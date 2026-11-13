"""Only verified new mappings may enter the glossary; user mappings are authoritative."""

import json
from unittest.mock import patch

import pytest
from wenyi_core.agents.glossary_extractor import GlossaryExtractor
from wenyi_core.agents.glossary_verifier import GlossaryVerificationError
from wenyi_core.config import Config
from wenyi_core.llm.operations import configured_operations
from wenyi_core.llm.providers.fake import FakeClient
from wenyi_core.models.glossary import GlossaryTerm
from wenyi_core.storage.file import FileStorage


def settings():
    return Config.from_dict(
        {
            "language": {"source": "en", "target": "zh"},
            "llm": {"preset": "fake"},
            "pipeline": {
                "glossary_verification": True,
                "align_retry_limit": 0,
                "glossary_fixed_terms": {"Protomolecule": "原分子"},
            },
        }
    )


def candidate_reply():
    return json.dumps(
        {
            "terms": [
                {"source": "Travon Barrish", "target": "特拉von·巴里什", "type": "person"},
                {"source": "Protomolecule", "target": "原原子物质"},
                {"source": "Known", "target": "已确认"},
            ]
        }
    )


def test_new_candidates_are_reviewed_before_storage_and_confirmed_terms_are_reused(tmp_path):
    cfg = settings()

    def handler(messages, tier, json_mode):
        if "verify terminology candidates" in messages[0]["content"]:
            evidence = json.loads(messages[1]["content"])
            assert [item["source"] for item in evidence] == ["Travon Barrish"]
            assert len(evidence[0]["source_context"]) < 600
            return '{"decisions":[{"id":"c0","accept":true,"target":"特拉文·巴里什"}]}'
        return candidate_reply()

    client = FakeClient(handler=handler, config=cfg.llm)
    store = FileStorage(str(tmp_path))
    store.upsert_term(GlossaryTerm(source="Known", target="已确认"))
    extractor = GlossaryExtractor(client, cfg)
    summary = extractor.extract_and_store(
        store,
        "Travon Barrish studied Protomolecule. Known.",
        "特拉von·巴里什研究原原子物质。已确认。",
        0,
    )
    assert store.get_term("Travon Barrish").target == "特拉文·巴里什"
    assert store.get_term("Protomolecule").target == "原分子"
    assert summary["inserted"] == 1
    assert [c["operation"] for c in client.calls] == ["glossary.extract", "glossary.verify"]
    assert client.calls[-1]["response_schema"]["properties"]["decisions"]["maxItems"] == 1
    store.close()


def test_verification_checkpoint_prevents_duplicate_inference_after_admission_failure(tmp_path):
    cfg = settings()

    def handler(messages, tier, json_mode):
        if "verify terminology candidates" in messages[0]["content"]:
            return '{"decisions":[{"id":"c0","accept":true,"target":"特拉文"}]}'
        return '{"terms":[{"source":"Travon","target":"特拉von"}]}'

    client = FakeClient(handler=handler, config=cfg.llm)
    store = FileStorage(str(tmp_path))
    extractor = GlossaryExtractor(client, cfg)
    extractor.seed_fixed_terms(store)
    with patch.object(store, "upsert_term", side_effect=RuntimeError("interrupted admission")):
        with pytest.raises(RuntimeError, match="interrupted admission"):
            extractor.extract_and_store(store, "Travon arrived.", "特拉von来了。", 0)
    extractor.extract_and_store(store, "Travon arrived.", "特拉von来了。", 0)
    assert sum(c["operation"] == "glossary.verify" for c in client.calls) == 1
    assert store.get_term("Travon").target == "特拉文"
    store.close()


def test_invalid_verification_does_not_admit_untrusted_candidates(tmp_path):
    cfg = settings()
    client = FakeClient(
        handler=lambda messages, *_: (
            '{"decisions":[]}'
            if "verify terminology candidates" in messages[0]["content"]
            else '{"terms":[{"source":"Travon","target":"特拉von"}]}'
        ),
        config=cfg.llm,
    )
    store = FileStorage(str(tmp_path))
    with pytest.raises(GlossaryVerificationError):
        GlossaryExtractor(client, cfg).extract_and_store(
            store, "Travon arrived.", "特拉von来了。", 0
        )
    assert store.get_term("Travon") is None
    assert store.get_term("Protomolecule").target == "原分子"
    store.close()


def test_credentials_only_include_verifier_when_enabled():
    cfg = settings()
    assert "glossary.verify" in configured_operations(cfg, "translate")
    cfg.pipeline.glossary_verification = False
    assert "glossary.verify" not in configured_operations(cfg, "translate")


def test_verification_is_reused_after_some_approved_candidates_were_already_admitted(tmp_path):
    cfg = settings()

    def handler(messages, tier, json_mode):
        if "verify terminology candidates" in messages[0]["content"]:
            evidence = json.loads(messages[1]["content"])
            return json.dumps(
                {
                    "decisions": [
                        {
                            "id": item["id"],
                            "accept": True,
                            "target": {"Travon": "特拉文", "Mira": "米拉"}[item["source"]],
                        }
                        for item in evidence
                    ]
                }
            )
        return (
            '{"terms":[{"source":"Travon","target":"特拉von"},{"source":"Mira","target":"Mira"}]}'
        )

    client = FakeClient(handler=handler, config=cfg.llm)
    store = FileStorage(str(tmp_path))
    extractor = GlossaryExtractor(client, cfg)
    extractor.seed_fixed_terms(store)
    upsert = store.upsert_term

    def interrupted_upsert(term, chapter=None):
        if term.source == "Mira":
            raise RuntimeError("interrupted second admission")
        return upsert(term, chapter=chapter)

    with patch.object(store, "upsert_term", side_effect=interrupted_upsert):
        with pytest.raises(RuntimeError, match="interrupted second"):
            extractor.extract_and_store(store, "Travon met Mira.", "特拉von遇到了Mira。", 0)
    assert store.get_term("Travon").target == "特拉文"
    assert store.get_term("Mira") is None
    extractor.extract_and_store(store, "Travon met Mira.", "特拉von遇到了Mira。", 0)
    assert sum(call["operation"] == "glossary.verify" for call in client.calls) == 1
    assert store.get_term("Mira").target == "米拉"
    store.close()


@pytest.mark.parametrize(
    "response",
    [
        '{"decisions":[{"id":"c0","accept":true,"target":特拉文}]}',
        '{"decisions":[{"id":"c0","accept":true,"target":"特拉文","target":"错误"}]}',
        '{"decisions":[{"id":"c0","accept":true,"target":"特拉文","extra":1}]}',
    ],
)
def test_malformed_or_schema_violating_verification_never_enters_glossary(tmp_path, response):
    cfg = settings()
    client = FakeClient(
        handler=lambda messages, *_: (
            response
            if "verify terminology candidates" in messages[0]["content"]
            else '{"terms":[{"source":"Travon","target":"特拉von"}]}'
        ),
        config=cfg.llm,
    )
    store = FileStorage(str(tmp_path))
    with pytest.raises(GlossaryVerificationError):
        GlossaryExtractor(client, cfg).extract_and_store(
            store, "Travon arrived.", "特拉von来了。", 0
        )
    assert store.get_term("Travon") is None
    store.close()


def test_substring_only_candidate_is_not_sent_for_verification_or_admitted(tmp_path):
    cfg = settings()
    client = FakeClient(
        handler=lambda messages, *_: (
            '{"decisions":[{"id":"c0","accept":true,"target":"安"}]}'
            if "verify terminology candidates" in messages[0]["content"]
            else '{"terms":[{"source":"Ann","target":"安"}]}'
        ),
        config=cfg.llm,
    )
    store = FileStorage(str(tmp_path))
    GlossaryExtractor(client, cfg).extract_and_store(
        store, "The annual ceremony began.", "年度典礼开始。", 0
    )
    assert store.get_term("Ann") is None
    assert all(call["operation"] != "glossary.verify" for call in client.calls)
    store.close()


def test_candidate_evidence_uses_a_real_word_occurrence_after_an_earlier_substring(tmp_path):
    cfg = settings()

    def handler(messages, tier, json_mode):
        if "verify terminology candidates" in messages[0]["content"]:
            evidence = json.loads(messages[1]["content"])
            assert "Ann arrived." in evidence[0]["source_context"]
            return '{"decisions":[{"id":"c0","accept":true,"target":"安"}]}'
        return '{"terms":[{"source":"Ann","target":"安"}]}'

    client = FakeClient(handler=handler, config=cfg.llm)
    store = FileStorage(str(tmp_path))
    source = "The annual ceremony began. " + "An unrelated sentence. " * 30 + "Ann arrived."
    GlossaryExtractor(client, cfg).extract_and_store(store, source, "安来了。", 0)
    assert store.get_term("Ann").target == "安"
    store.close()


def test_changed_user_mapping_invalidates_a_previous_candidate_decision(tmp_path):
    cfg = settings()

    def handler(messages, tier, json_mode):
        if "verify terminology candidates" in messages[0]["content"]:
            evidence = json.loads(messages[1]["content"])
            target = evidence[0]["existing_target"] or "特拉文"
            return json.dumps({"decisions": [{"id": "c0", "accept": True, "target": target}]})
        return '{"terms":[{"source":"Travon","target":"特拉von"}]}'

    client = FakeClient(handler=handler, config=cfg.llm)
    store = FileStorage(str(tmp_path))
    extractor = GlossaryExtractor(client, cfg)
    extractor.extract_and_store(store, "Travon arrived.", "特拉von来了。", 0)
    store.resolve_term("Travon", "用户指定译名")
    extractor.extract_and_store(store, "Travon arrived.", "特拉von来了。", 0)
    assert sum(call["operation"] == "glossary.verify" for call in client.calls) == 2
    assert store.get_term("Travon").target == "用户指定译名"
    store.close()


@pytest.mark.parametrize(
    "response",
    [
        '{"decisions":[{"id":"c0","accept":true,"target":"甲"},{"id":"c0","accept":true,"target":"乙"}]}',
        '{"decisions":[{"id":"c0","accept":true,"target":"甲"}]}',
    ],
)
def test_duplicate_or_missing_candidate_ids_never_publish_partial_results(tmp_path, response):
    cfg = settings()
    client = FakeClient(
        handler=lambda messages, *_: (
            response
            if "verify terminology candidates" in messages[0]["content"]
            else '{"terms":[{"source":"Travon","target":"甲"},{"source":"Mira","target":"乙"}]}'
        ),
        config=cfg.llm,
    )
    store = FileStorage(str(tmp_path))
    with pytest.raises(GlossaryVerificationError):
        GlossaryExtractor(client, cfg).extract_and_store(store, "Travon met Mira.", "甲遇见乙。", 0)
    assert store.get_term("Travon") is None
    assert store.get_term("Mira") is None
    store.close()
