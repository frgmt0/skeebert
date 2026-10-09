"""The discord.py adapter. The only module that imports discord.

It turns gateway events into calls on ``service.Skeebert`` and posts what comes
back. All behaviour (what is kept, scoring, tiers, tips, the dictionary) lives
in the service; this file only decides *which* messages reach it:

* a DM to Skeebert, or a guild message that @mentions Skeebert -> talk
* a reply to one of Skeebert's glyph messages -> a guess (scored, answered
  with a tier reaction only, so the meaning is not spoiled for others; a
  person's very first interaction also gets the one-line privacy notice)
* everything else, and anything from a bot -> ignored, never read further

Before any text reaches the service, ``clean_text`` replaces Discord markup
that carries ids (other people's mentions, roles, channels, custom emoji), so
no raw Discord id is stored or sent to Anthropic.

The Guess button is a ``DynamicItem`` whose custom_id carries the exchange id,
so buttons on old messages keep working after a restart.

Discord setup: enable the **Message Content** privileged intent in the
developer portal (Bot tab). Without it Discord still delivers the content of
DMs and of messages that mention Skeebert, but not of plain replies; such a
reply arrives empty and is ignored.
"""

from __future__ import annotations

import asyncio
import io
import logging
import re

import discord
from discord import app_commands

from .service import ExchangeGone, Skeebert, TalkReply

log = logging.getLogger(__name__)

GUESS_PREFIX = "skeebert:guess:"
MAX_SAY_CHARS = 1000
MAX_GUESS_CHARS = 200
GONE_TEXT = "that glyph's meaning is gone: its owner deleted it, or it was never kept and has expired."
OOPS_TEXT = "something went wrong on skeebert's side. try again in a bit."

_USER_MENTION = re.compile(r"<@!?(\d+)>")
_ROLE_MENTION = re.compile(r"<@&\d+>")
_CHANNEL_MENTION = re.compile(r"<#\d+>")
_CUSTOM_EMOJI = re.compile(r"<a?:(\w{1,32}):\d+>")


def clean_text(content: str, bot_user_id: int | None) -> str:
    """Strip Discord markup that carries ids before text reaches the service (and so the DB and Anthropic).

    Skeebert's own mention disappears; anyone else's becomes ``@someone``,
    roles ``@role``, channels ``#channel``, custom emoji ``:name:``.
    Whitespace is collapsed.
    """

    def user(m: re.Match) -> str:
        return "" if bot_user_id is not None and int(m.group(1)) == bot_user_id else "@someone"

    text = _USER_MENTION.sub(user, content or "")
    text = _ROLE_MENTION.sub("@role", text)
    text = _CHANNEL_MENTION.sub("#channel", text)
    text = _CUSTOM_EMOJI.sub(lambda m: f":{m.group(1)}:", text)
    return " ".join(text.split())


async def _report_failure(interaction: discord.Interaction, error: BaseException) -> None:
    """Log an interaction failure and tell the person, whatever state the response is in."""
    log.error("interaction failed", exc_info=error)
    try:
        if interaction.response.is_done():
            await interaction.followup.send(OOPS_TEXT, ephemeral=True)
        else:
            await interaction.response.send_message(OOPS_TEXT, ephemeral=True)
    except discord.HTTPException:
        log.warning("could not report the failure to the user")


def _file(png: bytes, name: str = "glyph.png") -> discord.File:
    return discord.File(io.BytesIO(png), filename=name)


class GuessModal(discord.ui.Modal, title="what did skeebert mean?"):
    def __init__(self, exchange_id: str) -> None:
        super().__init__(timeout=600)
        self.exchange_id = exchange_id
        self.guess = discord.ui.TextInput(
            label="your guess", placeholder="e.g. are you hungry?", max_length=MAX_GUESS_CHARS, required=True
        )
        self.add_item(self.guess)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        client: SkeebertBot = interaction.client  # type: ignore[assignment]
        await interaction.response.defer(ephemeral=True, thinking=True)
        text = clean_text(self.guess.value, client.user.id if client.user else None)[:MAX_GUESS_CHARS]
        try:
            outcome = await asyncio.to_thread(
                client.service.handle_guess, interaction.user.id, self.exchange_id, text, via="button"
            )
        except ExchangeGone:
            await interaction.followup.send(GONE_TEXT, ephemeral=True)
            return
        await interaction.followup.send(outcome.render(), ephemeral=True)
        if not outcome.already and interaction.message is not None:
            await client.refresh_counter(interaction.message, self.exchange_id)

    async def on_error(self, interaction: discord.Interaction, error: Exception) -> None:
        await _report_failure(interaction, error)


class GuessButton(discord.ui.DynamicItem[discord.ui.Button], template=GUESS_PREFIX + r"(?P<xid>[0-9a-f]{12})"):
    def __init__(self, exchange_id: str) -> None:
        super().__init__(
            discord.ui.Button(label="Guess", style=discord.ButtonStyle.primary, custom_id=GUESS_PREFIX + exchange_id)
        )
        self.exchange_id = exchange_id

    @classmethod
    async def from_custom_id(cls, interaction: discord.Interaction, item: discord.ui.Button, match: re.Match[str], /):
        return cls(match["xid"])

    async def callback(self, interaction: discord.Interaction) -> None:
        try:
            await self._callback(interaction)
        except Exception as exc:  # dynamic items have no view of ours to route errors through
            await _report_failure(interaction, exc)

    async def _callback(self, interaction: discord.Interaction) -> None:
        client: SkeebertBot = interaction.client  # type: ignore[assignment]
        try:
            prior = await asyncio.to_thread(client.service.existing_guess, interaction.user.id, self.exchange_id)
        except ExchangeGone:
            await interaction.response.send_message(GONE_TEXT, ephemeral=True)
            return
        if prior is not None:
            await interaction.response.send_message(prior.render(), ephemeral=True)
            return
        await interaction.response.send_modal(GuessModal(self.exchange_id))


def guess_view(exchange_id: str) -> discord.ui.View:
    view = discord.ui.View(timeout=None)
    view.add_item(GuessButton(exchange_id))
    return view


class ConfirmDeleteView(discord.ui.View):
    def __init__(self, owner_id: int) -> None:
        super().__init__(timeout=300)
        self.owner_id = owner_id

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        return interaction.user.id == self.owner_id

    async def on_error(self, interaction: discord.Interaction, error: Exception, item: discord.ui.Item) -> None:
        await _report_failure(interaction, error)

    @discord.ui.button(label="Yes, delete everything", style=discord.ButtonStyle.danger)
    async def confirm(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        client: SkeebertBot = interaction.client  # type: ignore[assignment]
        result = await asyncio.to_thread(client.service.delete_everything, interaction.user.id)
        await interaction.response.edit_message(
            content=(
                f"deleted: {result.exchanges} message{'s' if result.exchanges != 1 else ''} "
                f"(with {result.guesses_on_their_exchanges} guesses people made on them) and "
                f"{result.their_guesses} of your guesses. deleted data is excluded from all future training."
            ),
            view=None,
        )

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.secondary)
    async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await interaction.response.edit_message(content="nothing was deleted.", view=None)


class PrivacyView(discord.ui.View):
    def __init__(self, owner_id: int, keeping: bool) -> None:
        super().__init__(timeout=600)
        self.owner_id = owner_id
        self.toggle.label = "Stop keeping" if keeping else "Start keeping again"
        self.toggle.style = discord.ButtonStyle.secondary if keeping else discord.ButtonStyle.success

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        return interaction.user.id == self.owner_id

    async def on_error(self, interaction: discord.Interaction, error: Exception, item: discord.ui.Item) -> None:
        await _report_failure(interaction, error)

    @discord.ui.button(label="Stop keeping", style=discord.ButtonStyle.secondary)
    async def toggle(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        client: SkeebertBot = interaction.client  # type: ignore[assignment]
        current = await asyncio.to_thread(client.service.privacy_status, interaction.user.id)
        status = await asyncio.to_thread(client.service.set_keeping, interaction.user.id, not current.keeping)
        await interaction.response.edit_message(
            content=status.render(client.service.settings.privacy_url), view=PrivacyView(self.owner_id, status.keeping)
        )

    @discord.ui.button(label="Delete everything", style=discord.ButtonStyle.danger)
    async def delete(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await interaction.response.edit_message(
            content="delete every message, guess and glyph skeebert has kept from you? this can't be undone.",
            view=ConfirmDeleteView(self.owner_id),
        )


def build_command_group(bot: "SkeebertBot") -> app_commands.Group:
    group = app_commands.Group(name="skeebert", description="talk to skeebert, the alien who speaks in glyphs")

    @group.command(name="say", description="say something to skeebert; it answers with a glyph")
    @app_commands.describe(message="what you want to say")
    async def say(interaction: discord.Interaction, message: app_commands.Range[str, 1, MAX_SAY_CHARS]) -> None:
        await interaction.response.defer(thinking=True)
        key = f"channel:{interaction.channel_id}" if interaction.channel_id else f"user:{interaction.user.id}"
        text = clean_text(message, bot.user.id if bot.user else None)
        reply = await asyncio.to_thread(
            bot.service.handle_talk, interaction.user.id, text, source="slash", conversation_key=key
        )
        sent = await interaction.followup.send(
            content=reply.content, file=_file(reply.png), view=guess_view(reply.exchange_id), wait=True
        )
        await asyncio.to_thread(bot.service.attach_message, reply.exchange_id, sent.id)

    @group.command(name="dictionary", description="the glyphs people have cracked so far")
    async def dictionary(interaction: discord.Interaction) -> None:
        await interaction.response.defer(thinking=True)
        d = await asyncio.to_thread(bot.service.dictionary)
        text = d.render()[:1900]
        if d.png is not None:
            await interaction.followup.send(content=text, file=_file(d.png, "dictionary.png"))
        else:
            await interaction.followup.send(content=text)

    @group.command(name="help", description="how to talk to skeebert")
    async def help_(interaction: discord.Interaction) -> None:
        await interaction.response.send_message(bot.service.help(), ephemeral=True)

    @group.command(name="privacy", description="what skeebert keeps about you; stop keeping or delete it")
    async def privacy(interaction: discord.Interaction) -> None:
        status = await asyncio.to_thread(bot.service.privacy_status, interaction.user.id)
        await interaction.response.send_message(
            status.render(bot.service.settings.privacy_url),
            view=PrivacyView(interaction.user.id, status.keeping),
            ephemeral=True,
        )

    return group


class SkeebertBot(discord.Client):
    def __init__(self, service: Skeebert, *, dev_guild_id: int | None = None) -> None:
        intents = discord.Intents.default()
        intents.message_content = True  # privileged: enable it in the developer portal
        super().__init__(intents=intents, allowed_mentions=discord.AllowedMentions.none())
        self.service = service
        self.dev_guild_id = dev_guild_id
        self.tree = app_commands.CommandTree(self)
        self.tree.add_command(build_command_group(self))
        self.tree.on_error = self._on_app_command_error  # type: ignore[assignment]
        self._edit_locks: dict[int, asyncio.Lock] = {}
        self._warmup: asyncio.Task | None = None

    async def setup_hook(self) -> None:
        self.add_dynamic_items(GuessButton)
        if self.dev_guild_id:
            guild = discord.Object(id=self.dev_guild_id)
            self.tree.copy_global_to(guild=guild)
            synced = await self.tree.sync(guild=guild)
        else:
            synced = await self.tree.sync()
        log.info("synced %d application command(s)", len(synced))
        # Load the sentence model now rather than on the first guess (keeps the first guess fast).
        self._warmup = self.loop.create_task(self._warm_up())

    async def _warm_up(self) -> None:
        try:
            await asyncio.to_thread(self.service.embedder.encode, ["hello"])
            log.info("sentence model loaded")
        except Exception:
            log.exception("sentence model warm-up failed; it will be retried on the first guess")

    async def on_ready(self) -> None:
        log.info("ready as %s (model %s)", self.user, self.service.model_version)

    async def _on_app_command_error(self, interaction: discord.Interaction, error: app_commands.AppCommandError) -> None:
        await _report_failure(interaction, error)

    # -- messages -------------------------------------------------------------
    def clean(self, content: str) -> str:
        return clean_text(content, self.user.id if self.user else None)

    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot or self.user is None or message.author.id == self.user.id:
            return
        try:
            await self._route(message)
        except Exception:
            log.exception("handling a message failed")
            try:
                await message.reply(content=OOPS_TEXT, mention_author=False)
            except discord.HTTPException:
                pass

    async def _route(self, message: discord.Message) -> None:
        ref = message.reference
        if ref is not None and ref.message_id is not None:
            ex = await asyncio.to_thread(self.service.exchange_for_message, ref.message_id)
            if ex is not None:
                await self._reply_guess(message, ex.id, ref)
                return
        is_dm = message.guild is None
        mentioned = any(u.id == self.user.id for u in message.mentions)
        if not (is_dm or mentioned):
            return
        await self._talk(message, source="dm" if is_dm else "mention")

    async def _talk(self, message: discord.Message, *, source: str) -> None:
        text = self.clean(message.content or "")
        async with message.channel.typing():
            reply: TalkReply = await asyncio.to_thread(
                self.service.handle_talk, message.author.id, text, source=source,
                conversation_key=f"channel:{message.channel.id}",
            )
        sent = await message.reply(
            content=reply.content or None, file=_file(reply.png), view=guess_view(reply.exchange_id),
            mention_author=False,
        )
        await asyncio.to_thread(self.service.attach_message, reply.exchange_id, sent.id)

    async def _reply_guess(self, message: discord.Message, exchange_id: str, ref: discord.MessageReference) -> None:
        text = self.clean(message.content or "")[:MAX_GUESS_CHARS]
        if not text:
            return  # empty (or content intent off): nothing to score
        try:
            outcome = await asyncio.to_thread(
                self.service.handle_guess, message.author.id, exchange_id, text, via="reply"
            )
        except ExchangeGone:
            return
        if outcome.already:
            return  # only the first guess counts; later replies are ignored
        try:
            await message.add_reaction(outcome.tier_emoji)
        except discord.HTTPException as exc:
            log.warning("could not react to a guess: %s", exc)
        if outcome.notice:  # first time this person meets Skeebert: the notice only, never the reveal
            try:
                await message.reply(content=outcome.notice, mention_author=False)
            except discord.HTTPException as exc:
                log.warning("could not post the first-time notice: %s", exc)
        target = ref.resolved if isinstance(ref.resolved, discord.Message) else None
        if target is None:
            try:
                target = await message.channel.fetch_message(ref.message_id)  # type: ignore[arg-type]
            except discord.HTTPException:
                return
        await self.refresh_counter(target, exchange_id)

    def _edit_lock(self, message_id: int) -> asyncio.Lock:
        lock = self._edit_locks.get(message_id)
        if lock is None:
            if len(self._edit_locks) > 1000:  # drop idle locks
                for mid in [m for m, lk in self._edit_locks.items() if not lk.locked()]:
                    del self._edit_locks[mid]
            lock = self._edit_locks[message_id] = asyncio.Lock()
        return lock

    async def refresh_counter(self, glyph_message: discord.Message, exchange_id: str) -> None:
        """Rewrite the glyph message's text with the current 'decoded by N' counter.

        Edits of one message are serialised and the text is computed inside
        the lock, so a slower edit can never overwrite a newer count.
        """
        async with self._edit_lock(glyph_message.id):
            try:
                content = await asyncio.to_thread(self.service.message_content, exchange_id)
                await glyph_message.edit(content=content or None)
            except ExchangeGone:
                return
            except discord.HTTPException as exc:
                log.warning("could not update the guess counter: %s", exc)


def run(service: Skeebert, token: str, dev_guild_id: int | None = None) -> None:
    """Connect and run until interrupted. Requires a real bot token."""
    SkeebertBot(service, dev_guild_id=dev_guild_id).run(token, log_handler=None)
