from hsi_harness.answer_eval import NormalizationSection, f1, normalize

SQUAD = NormalizationSection(lowercase=True, strip_punctuation=True, remove_words=["a", "an", "the"])


def test_normalization_follows_the_configured_rules():
    assert normalize("The  Southampton, Hampshire!", SQUAD) == "southampton hampshire"
    keep = NormalizationSection(lowercase=False, strip_punctuation=False, remove_words=[])
    assert normalize("The X!", keep) == "The X!"


def test_token_f1_partial_and_exact():
    assert f1("Southampton, Hampshire", "Southampton", SQUAD) == 2 * (1 / 2 * 1) / (1 / 2 + 1)
    assert f1("the 2 February 1345", "2 February 1345", SQUAD) == 1.0
    assert f1("UK", "American", SQUAD) == 0.0
    assert f1("", "", SQUAD) == 1.0
