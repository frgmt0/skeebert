import discord
import pytest

from skeebert import bot
from skeebert.service import Skeebert


@pytest.fixture
def service(settings, engine, embedder):
    from skeebert.brain import LocalBrain
    from skeebert.identity import Pseudonymiser
    from skeebert.store import Store

    return Skeebert(settings, Store(settings.db_path), Pseudonymiser.load(settings.salt_path), engine, embedder,
                    LocalBrain(embedder))


def test_bot_builds_command_tree_without_connecting(service):
    client = bot.SkeebertBot(service)
    [group] = client.tree.get_commands()
    assert group.name == "skeebert"
    assert sorted(c.name for c in group.commands) == ["dictionary", "help", "privacy", "say"]
    assert client.intents.message_content and client.intents.dm_messages and client.intents.guild_messages
    assert client.allowed_mentions.everyone is False and client.allowed_mentions.users is False
    assert not client.is_ready()


def test_guess_button_custom_id_is_persistent():
    item = bot.GuessButton("0123456789ab")
    assert item.custom_id == "skeebert:guess:0123456789ab"
    match = bot.GuessButton.__discord_ui_compiled_template__.fullmatch(item.custom_id)
    assert match["xid"] == "0123456789ab"
    view = bot.guess_view("0123456789ab")
    assert view.timeout is None and view.is_persistent()


def test_privacy_view_labels():
    assert bot.PrivacyView(1, True).toggle.label == "Stop keeping"
    assert bot.PrivacyView(1, False).toggle.label == "Start keeping again"
    labels = [c.label for c in bot.PrivacyView(1, True).children]
    assert "Delete everything" in labels


def test_clean_text_removes_every_discord_id():
    text = "<@42> hey <@!42> ask <@7> and <@!123456789012345678> in <#555> cc <@&999> <:blob:12345> <a:dance:678>"
    out = bot.clean_text(text, 42)
    assert out == "hey ask @someone and @someone in #channel cc @role :blob: :dance:"
    assert not any(ch.isdigit() for ch in out)
    assert bot.clean_text("<@42>", 42) == ""
    assert bot.clean_text("<@42> hi", None) == "@someone hi"
    assert bot.clean_text("plain 123 numbers stay", 42) == "plain 123 numbers stay"


def test_client_clean_uses_own_id(service):
    client = bot.SkeebertBot(service)
    client._connection.user = type("U", (), {"id": 42})()
    assert client.clean("<@42> hey <@7>") == "hey @someone"


# -- message routing with stand-in gateway objects (no connection) -------------

class _Typing:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class FakeChannel:
    def __init__(self, cid=500):
        self.id = cid
        self.fetched = []

    def typing(self):
        return _Typing()

    async def fetch_message(self, mid):
        self.fetched.append(mid)
        return FakeSent(mid)


class FakeSent:
    def __init__(self, mid):
        self.id = mid
        self.edits = []

    async def edit(self, **kw):
        self.edits.append(kw)


_ids = iter(range(9000, 10**6))


class FakeMessage:
    def __init__(self, content, author_id=10, *, bot_author=False, guild=True, mentions=(), reference=None):
        self.content = content
        self.author = type("A", (), {"id": author_id, "bot": bot_author})()
        self.guild = object() if guild else None
        self.mentions = [type("M", (), {"id": m})() for m in mentions]
        self.reference = reference
        self.channel = FakeChannel()
        self.replies = []
        self.reactions = []

    async def reply(self, **kw):
        sent = FakeSent(next(_ids))  # Discord message ids are unique
        self.replies.append((kw, sent))
        return sent

    async def add_reaction(self, emoji):
        self.reactions.append(emoji)


@pytest.fixture
def client(service):
    c = bot.SkeebertBot(service)
    c._connection.user = type("U", (), {"id": 42})()
    return c


async def test_ignores_unaddressed_and_bots(client, service):
    for msg in (FakeMessage("hello all"), FakeMessage("<@42> hi", bot_author=True, mentions=[42])):
        await client.on_message(msg)
        assert msg.replies == []
    assert service.store.summary()["exchanges"] == 0


async def test_mention_and_dm_talk(client, service):
    m = FakeMessage("<@42> are you hungry?", mentions=[42])
    await client.on_message(m)
    [(kw, sent)] = m.replies
    assert isinstance(kw["file"], discord.File) and kw["mention_author"] is False
    assert service.exchange_for_message(sent.id) is not None
    rows = service.store._all("SELECT message_text, source FROM exchanges")
    assert [(r[0], r[1]) for r in rows] == [("are you hungry?", "mention")]
    other = FakeMessage("<@42> is <@!31337> <#77> <@&5> hungry?", mentions=[42, 31337])
    await client.on_message(other)
    stored = [r[0] for r in service.store._all("SELECT message_text FROM exchanges")]
    assert stored[-1] == "is @someone #channel @role hungry?"
    dm = FakeMessage("", guild=False)  # empty content is handled
    await client.on_message(dm)
    assert len(dm.replies) == 1


async def test_reply_to_glyph_is_a_guess(client, service):
    m = FakeMessage("<@42> food", mentions=[42])
    await client.on_message(m)
    [(_, glyph_msg)] = m.replies
    ref = discord.MessageReference(message_id=glyph_msg.id, channel_id=500)
    await client.on_message(FakeMessage("<@42> hi", author_id=11, mentions=[42]))  # 11 has seen the notice
    guess = FakeMessage("<@42> food please", author_id=11, mentions=[42], reference=ref)
    await client.on_message(guess)
    assert guess.replies == []  # no public reveal
    assert len(guess.reactions) == 1 and guess.reactions[0] in "🎯🔥🤏🧊"
    assert guess.channel.fetched == [glyph_msg.id]
    again = FakeMessage("something else", author_id=11, reference=ref)
    await client.on_message(again)
    assert again.reactions == []  # later replies by the same person are ignored
    assert service.store.summary()["guesses"] == 1


async def test_reply_to_unknown_message_with_mention_is_talk(client, service):
    ref = discord.MessageReference(message_id=123456, channel_id=500)
    m = FakeMessage("<@42> hi", mentions=[42], reference=ref)
    await client.on_message(m)
    assert len(m.replies) == 1 and m.reactions == []


async def test_first_reply_guess_gets_only_the_notice(client, service, settings):
    m = FakeMessage("<@42> food", mentions=[42])
    await client.on_message(m)
    [(_, glyph_msg)] = m.replies
    ref = discord.MessageReference(message_id=glyph_msg.id, channel_id=500)
    newcomer = FakeMessage("food " + "x" * 500, author_id=77, reference=ref)
    await client.on_message(newcomer)
    [(kw, _)] = newcomer.replies
    assert kw["content"] == service_first_line(settings) and "meant" not in kw["content"]
    assert len(newcomer.reactions) == 1
    uh = service.user_hash(77)
    assert service.store.get_user(uh).onboarded
    [g] = service.store._all("SELECT guess_text FROM guesses WHERE user_hash = ?", (uh,))
    assert len(g[0]) == bot.MAX_GUESS_CHARS  # reply guesses are clipped like the modal
    # a second newcomer guess elsewhere gets no further notice
    m2 = FakeMessage("<@42> water", mentions=[42])
    await client.on_message(m2)
    ref2 = discord.MessageReference(message_id=m2.replies[0][1].id, channel_id=500)
    again = FakeMessage("water", author_id=77, reference=ref2)
    await client.on_message(again)
    assert again.replies == []


def service_first_line(settings):
    from skeebert.service import first_time_line

    return first_time_line(settings.privacy_url)


async def test_on_message_failure_is_reported_not_raised(client, service, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("kaput")

    monkeypatch.setattr(service, "handle_talk", boom)
    m = FakeMessage("<@42> hi", mentions=[42])
    await client.on_message(m)
    [(kw, _)] = m.replies
    assert kw["content"] == bot.OOPS_TEXT


async def test_counter_edits_are_serialised_and_fresh(client, service):
    import asyncio

    m = FakeMessage("<@42> food", mentions=[42])
    await client.on_message(m)
    [(_, glyph_msg)] = m.replies
    xid = service.exchange_for_message(glyph_msg.id).id
    service.handle_guess(5, xid, "food", via="reply")
    service.handle_guess(6, xid, "food", via="reply")
    await asyncio.gather(client.refresh_counter(glyph_msg, xid), client.refresh_counter(glyph_msg, xid))
    assert [e["content"].splitlines()[0] for e in glyph_msg.edits] == ["💤 · decoded by 2"] * 2


class _FakeResponse:
    def __init__(self):
        self.sent = []
        self._done = False

    def is_done(self):
        return self._done

    async def send_message(self, content, ephemeral=False, **kw):
        self.sent.append(content)
        self._done = True


async def test_guess_button_failure_is_reported(client, service, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("db exploded")

    monkeypatch.setattr(service, "existing_guess", boom)
    response = _FakeResponse()
    interaction = type("I", (), {"client": client, "response": response, "user": type("U", (), {"id": 3})()})()
    await bot.GuessButton("0123456789ab").callback(interaction)
    assert response.sent == [bot.OOPS_TEXT]
