"""HY-MT structured translation contracts, using offline model responses."""

import json

import pytest
from wenyi_core.agents.polisher import Polisher
from wenyi_core.agents.title_translator import TitleTranslator
from wenyi_core.agents.translator import AlignmentError, Translator
from wenyi_core.config import Config
from wenyi_core.llm.operations import configured_operations
from wenyi_core.llm.providers.fake import FakeClient
from wenyi_core.llm.routing import resolve_routes
from wenyi_core.models.glossary import GlossaryTerm
from wenyi_core.pipeline.translate.translation_batch import BatchPlan, TranslationBatchExecutor


def hy_config(**pipeline):
    return Config.from_dict(
        {
            "language": {"source": "en", "target": "zh"},
            "llm": {
                "preset": "fake",
                "models": {
                    "hy": {
                        "provider": "default",
                        "model": "translation-model",
                        "translation_protocol": "hy_mt",
                    }
                },
                "routes": {
                    "translation.body": {"model": "hy"},
                    "translation.title": {"model": "hy"},
                },
            },
            "pipeline": {"align_retry_limit": 0, **pipeline},
        }
    )


def test_body_uses_single_user_and_restores_key_order_without_polish_transcript():
    cfg = hy_config()
    client = FakeClient(handler=lambda *_: '{"p1":"第二段","p0":"第一段"}', config=cfg.llm)
    translator = Translator(client, cfg)
    assert translator.translate_batch(["First paragraph.", "Second paragraph."]) == [
        "第一段",
        "第二段",
    ]
    call = client.calls[0]
    assert [message["role"] for message in call["messages"]] == ["user"]
    assert not call["json_mode"]
    assert "Chinese" in call["messages"][0]["content"]
    assert '"p0": "First paragraph."' in call["messages"][0]["content"]
    assert translator.last_batch_turn is None


@pytest.mark.parametrize(
    "raw",
    [
        '{"p0":"a","p0":"b"}',
        '{"p1":"a"}',
        '{"p0":"a","p1":"extra"}',
        '{"p0":1}',
        '{"p0":""}',
        '```json\n{"p0":"a"}\n```',
        '{"p0":"a"} trailing',
    ],
)
def test_invalid_body_output_cannot_be_saved(raw):
    cfg = hy_config()
    client = FakeClient(handler=lambda *_: raw, config=cfg.llm)
    with pytest.raises(AlignmentError):
        Translator(client, cfg).translate_batch(["Source paragraph."])


def test_batch_failure_uses_existing_single_paragraph_recovery():
    cfg = hy_config()
    responses = iter(['{"p0":"incomplete"}', '{"p0":"甲"}', '{"p0":"乙"}'])
    client = FakeClient(handler=lambda *_: next(responses), config=cfg.llm)
    assert Translator(client, cfg).translate_batch(["One.", "Two."]) == ["甲", "乙"]
    assert len(client.calls) == 3


def test_changed_single_paragraph_key_uses_plain_translation_as_final_recovery():
    cfg = hy_config()
    responses = iter(['{"wrong":"invalid"}', '{"p2":"wrong key"}', "正确的译文。"])
    client = FakeClient(handler=lambda *_: next(responses), config=cfg.llm)
    assert Translator(client, cfg).translate_batch(["A translated paragraph."]) == ["正确的译文。"]
    assert len(client.calls) == 3
    assert "### Source Data" not in client.calls[-1]["messages"][0]["content"]


def test_invalid_polishing_stops_batch_publication_instead_of_silently_keeping_draft():
    from wenyi_core.agents.polisher import PolishOutputError

    cfg = hy_config()

    def respond(messages, tier, json_mode):
        return "[0]" if json_mode else '{"p0":"初译"}'

    client = FakeClient(handler=respond, config=cfg.llm)
    plan = BatchPlan(
        chapter=0,
        start_index=0,
        segment_indices=(0,),
        sources=("Source.",),
        terms=(),
        context="",
        style="",
        book_synopsis="",
        chapter_digest="",
        annotation_contexts=[[]],
        next_source="",
    )
    with pytest.raises(PolishOutputError):
        TranslationBatchExecutor(Translator(client, cfg), Polisher(client, cfg)).execute(
            plan, polish=True
        )


def test_fresh_polishing_receives_original_source_without_terminology_experiment():
    cfg = hy_config()
    client = FakeClient(handler=lambda *_: '{"polished":["修改后的译文"]}', config=cfg.llm)
    Polisher(client, cfg).polish(["有误的初译"], sources=["The original source meaning."])
    assert "The original source meaning." in client.calls[0]["messages"][-1]["content"]


def test_titles_use_the_same_structured_protocol():
    cfg = hy_config()
    client = FakeClient(handler=lambda *_: '{"p1":"雨中","p0":"蓝门"}', config=cfg.llm)
    assert TitleTranslator(client, cfg).translate(["Blue Door", "Rain"], "") == ["蓝门", "雨中"]
    assert client.calls[0]["operation"] == "translation.title"
    assert not client.calls[0]["json_mode"]


def test_protocol_is_visible_and_changes_only_selected_inference_identities():
    cfg = hy_config()
    hy_routes = resolve_routes(cfg.llm)
    raw = cfg.llm.model_dump()
    raw["models"]["hy"]["translation_protocol"] = "standard"
    standard = cfg.llm.model_validate(raw)
    standard_routes = resolve_routes(standard)
    assert hy_routes["translation.body"].describe()["translation_protocol"] == "hy_mt"
    assert (
        hy_routes["translation.body"].fingerprint != standard_routes["translation.body"].fingerprint
    )
    assert hy_routes["review.scan"].fingerprint == standard_routes["review.scan"].fingerprint


def test_precision_rejects_hy_mt_before_inference():
    with pytest.raises(ValueError, match="hy_mt.*three_draft_synthesis"):
        hy_config(translation_mode="three_draft_synthesis")


def test_mode_override_is_checked_before_workflow_inference():
    cfg = hy_config()
    cfg.pipeline.translation_mode = "three_draft_synthesis"
    with pytest.raises(ValueError, match="hy_mt.*three_draft_synthesis"):
        configured_operations(cfg, "translate")


@pytest.mark.parametrize("kind", ["wrong_operation", "mixed_fallback", "unknown_protocol"])
def test_unsupported_protocol_routes_fail_at_configuration_boundary(kind):
    raw = hy_config().llm.model_dump()
    if kind == "wrong_operation":
        raw["routes"]["synopsis.chapter"] = {"model": "hy"}
    elif kind == "mixed_fallback":
        raw["routes"]["translation.body"]["fallbacks"] = ["default_strong"]
    else:
        raw["models"]["hy"]["translation_protocol"] = "unknown"
    with pytest.raises(ValueError):
        hy_config().llm.model_validate(raw)


def test_context_stays_outside_source_json_and_prompt_is_frozen():
    cfg = hy_config()
    cfg.freeze_language_policies()
    assert "hy_mt_translation" in dict(cfg.language_policy().templates)
    client = FakeClient(handler=lambda *_: '{"p0":"米拉出门了。"}', config=cfg.llm)
    Translator(client, cfg).translate_batch(
        ["Mira left."],
        glossary_terms=[GlossaryTerm(source="Mira", target="米拉")],
        style="Quiet prose.",
        context="Earlier translation.",
        book_synopsis="A mystery.",
        chapter_digest="A departure.",
        annotation_contexts=[[{"target_key": "note1", "source": "A note."}]],
        next_source="A cat waits outside.",
    )
    content = client.calls[0]["messages"][0]["content"]
    assert json.loads(content.split("### Source Data\n")[1]) == {"p0": "Mira left."}
    for reference in (
        "Quiet prose.",
        "Earlier translation.",
        "A mystery.",
        "A departure.",
        "米拉",
        "A note.",
        "A cat waits outside.",
    ):
        assert reference in content


@pytest.mark.parametrize("genre", ["literary", "academic"])
def test_hy_mt_protocol_supports_runtime_prompt_genres(genre):
    cfg = hy_config()
    cfg.set_prompt_genre(genre)
    cfg.freeze_language_policies()
    assert cfg.language_policy().context.genre == genre
    assert "hy_mt_translation" in dict(cfg.language_policy().templates)
    client = FakeClient(handler=lambda *_: '{"p0":"译文"}', config=cfg.llm)
    assert Translator(client, cfg).translate_batch(["Source."]) == ["译文"]
    content = client.calls[0]["messages"][0]["content"]
    assert json.loads(content.split("### Source Data\n")[1]) == {"p0": "Source."}
    assert "${genre_" not in content


def test_filtered_sources_and_intentionally_empty_targets_keep_their_positions():
    cfg = hy_config()
    client = FakeClient(handler=lambda *_: '{"p0":""}', config=cfg.llm)
    assert Translator(client, cfg).translate_batch(
        ["-", "OCR junk", "42"], allow_empty_translations=True
    ) == ["-", "", "42"]
    content = client.calls[0]["messages"][0]["content"]
    assert json.loads(content.split("### Source Data\n")[1]) == {"p0": "OCR junk"}


def test_hy_translation_uses_fresh_polishing_and_preserves_before_polish():
    cfg = hy_config()
    responses = iter(['{"p0":"初稿"}', '{"polished":["润色稿"]}'])
    client = FakeClient(handler=lambda *_: next(responses), config=cfg.llm)
    plan = BatchPlan(
        chapter=0,
        start_index=0,
        segment_indices=(0,),
        sources=("Source.",),
        terms=(),
        context="",
        style="",
        book_synopsis="",
        chapter_digest="",
        annotation_contexts=[[]],
        next_source="",
    )
    result = TranslationBatchExecutor(Translator(client, cfg), Polisher(client, cfg)).execute(
        plan, polish=True
    )
    assert result.targets == ("润色稿",)
    assert result.before_polish == ("初稿",)
    assert [call["operation"] for call in client.calls] == ["translation.body", "polish.body"]
    assert [message["role"] for message in client.calls[1]["messages"]] == ["system", "user"]
