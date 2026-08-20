"""Test suite for the nl2ms measurement instrument.

Two kinds of test live here:

*Correctness tests* check metric definitions against closed-form values or
independent implementations (scipy). If entropy or JSD is wrong, every
downstream conclusion is wrong, so these are checked against references
rather than against previously-recorded outputs.

*Discrimination tests* check that the detection machinery says "no" when
there is nothing to find. An instrument that reports a critical transition in
structureless noise is worse than no instrument, so the null behaviour is
tested as carefully as the positive behaviour.

Run with ``python -m pytest tests/ -v`` or ``python tests/test_nl2ms.py``.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nl2ms import signals as S
from nl2ms import geometry as G
from nl2ms import stats as ST
from nl2ms import critical as CR
from nl2ms import storage as STO
from nl2ms.config import ExperimentConfig, smoke_config
from nl2ms.datasets_build import (ANSWER_CLOSED_SET, ANSWER_UNDEFINED,
                                  generate_synthetic, load_samples,
                                  make_sample_id, save_samples,
                                  stratified_subset)

TOL = 1e-9


# ===========================================================================
# Signals: entropy, divergence
# ===========================================================================
def test_entropy_closed_form():
    p_unif = np.ones(8) / 8
    assert abs(S.shannon_entropy(p_unif) - np.log(8)) < TOL
    p_peak = np.zeros(8); p_peak[0] = 1.0
    assert abs(S.shannon_entropy(p_peak)) < TOL
    assert abs(S.normalised_entropy(p_unif) - 1.0) < TOL
    assert abs(S.normalised_entropy(p_peak)) < TOL


def test_entropy_matches_scipy():
    from scipy.stats import entropy as sp_entropy
    rng = np.random.default_rng(0)
    for _ in range(10):
        p = rng.dirichlet(np.ones(20))
        assert abs(S.shannon_entropy(p) - sp_entropy(p)) < 1e-12


def test_kl_matches_scipy_and_is_inf_off_support():
    from scipy.stats import entropy as sp_entropy
    rng = np.random.default_rng(1)
    p, q = rng.dirichlet(np.ones(10)), rng.dirichlet(np.ones(10))
    assert abs(S.kl_divergence(p, q) - sp_entropy(p, q)) < 1e-12
    a = np.array([1.0, 0, 0, 0]); b = np.array([0, 1.0, 0, 0])
    # KL is genuinely infinite here; reporting a finite number would hide a
    # meaningless comparison.
    assert np.isinf(S.kl_divergence(a, b))


def test_jsd_properties():
    from scipy.spatial.distance import jensenshannon
    rng = np.random.default_rng(2)
    p, q = rng.dirichlet(np.ones(12)), rng.dirichlet(np.ones(12))
    assert abs(S.js_divergence(p, p)) < TOL                      # identity
    assert abs(S.js_divergence(p, q) - S.js_divergence(q, p)) < TOL   # symmetry
    ref = jensenshannon(p, q, base=np.e) ** 2
    assert abs(S.js_divergence(p, q) - ref) < 1e-12
    a = np.array([1.0, 0, 0, 0]); b = np.array([0, 1.0, 0, 0])
    assert abs(S.js_divergence(a, b) - np.log(2)) < 1e-12         # bounded
    assert S.js_divergence(p, q) <= np.log(2) + 1e-12


def test_layerwise_jsd_alignment():
    P = np.array([[.9, .05, .05], [.5, .3, .2], [.1, .8, .1]])
    out = S.layerwise_jsd(P)
    # Padded so index 0 is NaN-free zero and index l holds JSD(l-1, l).
    assert out["jsd_consecutive"].size == P.shape[0]
    assert out["jsd_consecutive"][0] == 0.0
    assert abs(out["jsd_consecutive"][1] - S.js_divergence(P[0], P[1])) < TOL
    assert np.all(np.diff(out["jsd_cumulative"]) >= -TOL)


# ===========================================================================
# Signals: order parameter and symmetry breaking
# ===========================================================================
def test_order_parameter_tie_and_dominance():
    cand = np.array([[.25, .25, .25, .25],   # perfect tie
                     [.40, .30, .20, .10],
                     [.90, .05, .03, .02]])  # dominant
    op = S.order_parameter_closed_set(cand, correct_index=0)
    assert abs(op["margin"][0]) < TOL                    # tie -> zero margin
    assert op["margin"][2] > op["margin"][1] > op["margin"][0]
    assert abs(op["symmetry_breaking_index"][0]) < TOL   # uniform -> SB = 0
    assert op["symmetry_breaking_index"][2] > 0.5
    assert list(op["rank"]) == [1, 1, 1]


def test_order_parameter_negative_when_wrong():
    cand = np.array([[.1, .7, .1, .1]])
    op = S.order_parameter_closed_set(cand, correct_index=0)
    assert op["margin"][0] < 0                # correct answer is losing
    assert op["rank"][0] == 2


def test_gini_reference_values():
    assert abs(S.gini_concentration(np.ones(10) / 10)) < 1e-9
    delta = np.zeros(10); delta[0] = 1.0
    assert abs(S.gini_concentration(delta) - 0.9) < 1e-9   # (n-1)/n


# ===========================================================================
# Signals: trajectory
# ===========================================================================
def test_straight_line_has_zero_curvature():
    H = np.stack([np.array([float(i), 0.0, 0.0]) for i in range(6)])
    t = S.trajectory_metrics(H)
    assert np.allclose(t["velocity"][1:], 1.0)
    assert np.allclose(t["curvature"], 0.0, atol=1e-9)
    assert np.allclose(t["turning_angle"], 0.0, atol=1e-9)


def test_right_angle_turn_detected():
    H = np.array([[0., 0.], [1., 0.], [1., 1.], [1., 2.]])
    t = S.trajectory_metrics(H)
    assert abs(t["turning_angle"][1] - np.pi / 2) < 1e-9
    assert t["turning_angle"][2] < 1e-9


def test_velocity_normalisation_removes_scale_growth():
    # A trajectory whose steps grow purely because its norm grows should have
    # roughly constant *normalised* velocity -- the property that makes the
    # normalised measure the trustworthy one.
    base = np.array([1.0, 0.0, 0.0])
    H = np.stack([base * (1.5 ** i) for i in range(8)])
    t = S.trajectory_metrics(H)
    raw = t["velocity"][2:]
    norm = t["velocity_normalised"][2:]
    assert raw.max() / raw.min() > 5           # raw velocity inflates a lot
    assert norm.max() / norm.min() < 1.01      # normalised does not


def test_path_length_monotone():
    rng = np.random.default_rng(3)
    H = np.cumsum(rng.normal(size=(10, 5)), axis=0)
    t = S.trajectory_metrics(H)
    assert np.all(np.diff(t["path_length"]) >= -TOL)
    assert abs(t["displacement_from_last"][-1]) < TOL


# ===========================================================================
# Signals: peak detection
# ===========================================================================
def test_transition_sharpness_and_interval():
    spike = np.array([0, 0, 1, 5, 1, 0, 0.])
    s = S.transition_sharpness(spike)
    assert s["peak_index"] == 3
    assert s["width_half_max"] == 1
    assert S.detect_interval(spike) == (3, 3)

    plateau = np.array([0, 1, 4, 5, 5, 4, 1, 0.])
    sp = S.transition_sharpness(plateau)
    assert sp["width_half_max"] >= 4          # broad, not a spike
    lo, hi = S.detect_interval(plateau)
    assert hi - lo >= 3


def test_safe_stack_rejects_mismatched_lengths():
    assert S.safe_stack([np.zeros(5), np.zeros(5)]).shape == (2, 5)
    # Mixing profiles of different depth means mixing models; padding them
    # would fabricate a comparison, so this must refuse.
    assert S.safe_stack([np.zeros(5), np.zeros(7)]) is None


# ===========================================================================
# Geometry
# ===========================================================================
def test_effective_rank_recovers_true_rank():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(200, 3)) @ rng.normal(size=(3, 20))
    er = G.effective_rank(G.singular_spectrum(X)["singular_values"])
    assert 2.5 < er < 3.5
    assert G.components_for_variance(
        G.singular_spectrum(X)["explained_variance_ratio"], 0.9) == 3


def test_effective_rank_full_for_isotropic():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(500, 10))
    er = G.effective_rank(G.singular_spectrum(X)["singular_values"])
    assert er > 9.0          # near-flat spectrum -> near-full effective rank


def test_intrinsic_dimension_estimate():
    rng = np.random.default_rng(6)
    Y = rng.uniform(size=(600, 2)) @ rng.normal(size=(2, 12))
    d = G.two_nn_intrinsic_dimension(Y)["intrinsic_dimension"]
    assert 1.5 < d < 3.0     # true dimension is 2


def test_neighbourhood_stability_bounds():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(60, 6))
    assert G.neighbourhood_stability(X, X, k=5)["knn_jaccard"] == 1.0
    scrambled = rng.normal(size=(60, 6))
    assert G.neighbourhood_stability(X, scrambled, k=5)["knn_jaccard"] < 0.3
    # Translation and rescaling preserve neighbour identity; the measure must
    # be blind to them, which is why it is used to detect *reorganisation*.
    assert G.neighbourhood_stability(X, X * 3.0 + 10.0, k=5)["knn_jaccard"] == 1.0


def test_class_separation_ignores_ungraded():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(30, 6)); X[:10] += 6.0
    labels = np.array([True] * 10 + [False] * 10 + [None] * 10, dtype=object)
    out = G.class_separation(X, labels)
    assert out["n_classes"] == 2
    assert out["n_labelled"] == 20        # the 10 ungraded are excluded
    assert out["separation_ratio"] > 1.5


# ===========================================================================
# Statistics
# ===========================================================================
def test_effect_sizes_against_reference():
    rng = np.random.default_rng(9)
    a = rng.normal(0, 1, 200)
    b = rng.normal(0.8, 1, 200)
    assert 0.6 < ST.hedges_g(b, a) < 1.0
    brute = np.mean([np.sign(u - v) for u in b[:60] for v in a[:60]])
    assert abs(ST.cliffs_delta(b[:60], a[:60]) - brute) < 1e-9


def test_hedges_g_smaller_than_cohens_d():
    rng = np.random.default_rng(10)
    a, b = rng.normal(0, 1, 8), rng.normal(1, 1, 8)
    assert abs(ST.hedges_g(a, b)) < abs(ST.cohens_d(a, b))


def test_benjamini_hochberg_reference():
    tests = [{"p_value": p} for p in [0.001, 0.008, 0.039, 0.041, 0.042, 0.6]]
    out = ST.correct_multiple_comparisons(tests, alpha=0.05)
    # BH: p_(i) * n / i, then enforce monotonicity.
    assert abs(out[0]["p_value_corrected"] - 0.006) < 1e-9
    assert abs(out[1]["p_value_corrected"] - 0.024) < 1e-9
    assert out[5]["significant"] is False
    assert all(r["n_tests_in_family"] == 6 for r in out)


def test_uncomputable_tests_keep_none_not_false():
    out = ST.correct_multiple_comparisons(
        [{"p_value": 0.01}, {"p_value": float("nan")}, {}])
    assert out[0]["significant"] is True
    # A test that could not run must not be silently recorded as
    # "not significant" -- that would understate the missing analysis.
    assert out[1]["significant"] is None
    assert out[2]["significant"] is None


def test_permutation_p_value_has_honest_floor():
    rng = np.random.default_rng(11)
    a, b = rng.normal(0, 1, 40), rng.normal(10, 1, 40)
    res = ST.permutation_test(a, b, n_perm=99)
    assert res["p_value"] == 1.0 / 100          # never exactly zero
    assert res["p_value_floor"] == 1.0 / 100


def test_bootstrap_ci_brackets_truth():
    rng = np.random.default_rng(12)
    x = rng.normal(5.0, 1.0, 300)
    ci = ST.bootstrap_ci(x, n_boot=800, seed=0)
    assert ci["ci_low"] < 5.0 < ci["ci_high"]
    assert ci["ci_low"] < ci["point"] < ci["ci_high"]


def test_compare_groups_refuses_tiny_groups():
    cfg = ExperimentConfig().stats
    out = ST.compare_groups([1.0, 2.0], [3.0, 4.0], cfg=cfg)
    assert out["status"] == "below_min_group_size"
    assert "effect_size" not in out          # nothing reported, not a fake zero


# ===========================================================================
# Null models: the instrument must say "no" to noise
# ===========================================================================
def test_layer_shuffle_null_is_calibrated_under_h0():
    """p-values must be roughly uniform when there is no structure.

    Checked across many draws rather than one: a single structureless draw
    legitimately produces p < 0.05 about 5% of the time, so asserting on one
    seed would test the seed, not the null model.
    """
    ps = []
    for s in range(24):
        rng = np.random.default_rng(1000 + s)
        noise = np.abs(rng.normal(size=(30, 33)))
        ps.append(ST.null_random_layer(noise, n_perm=150, seed=s)["p_value"])
    ps = np.array(ps)
    assert 0.3 < ps.mean() < 0.7          # uniform-ish, centred near 0.5
    assert (ps < 0.05).mean() <= 0.15     # false-positive rate near nominal


def test_layer_shuffle_null_detects_planted_peak():
    for s in range(5):
        rng = np.random.default_rng(2000 + s)
        curves = np.abs(rng.normal(size=(30, 33)))
        curves[:, 16] += 8.0               # a real, localised peak
        res = ST.null_random_layer(curves, n_perm=150, seed=s)
        assert res["exceeds_null"] is True
        assert res["p_value"] < 0.05


def test_label_shuffle_null_discriminates():
    rng = np.random.default_rng(14)
    values = np.concatenate([rng.normal(0, 1, 40), rng.normal(3, 1, 40)])
    labels = [0] * 40 + [1] * 40
    real = ST.null_label_shuffle(values, labels, n_perm=300, seed=0)
    assert real["exceeds_null"] is True

    shuffled_labels = list(rng.permutation(labels))
    fake = ST.null_label_shuffle(values, shuffled_labels, n_perm=300, seed=0)
    assert fake["p_value"] > 0.05


def test_detector_false_positive_rate_is_measured_not_assumed():
    res = ST.detector_false_positive_rate(40, 33, seed=0, n_repeats=8)
    assert res["status"] == "ok"
    assert 0.0 <= res["false_sharp_rate"] <= 1.0
    # At realistic depth, structureless AR(1) curves should rarely be called
    # sharp; if this ever rises, the shape classifier has become permissive.
    assert res["false_sharp_rate"] < 0.15


# ===========================================================================
# Critical-layer detection
# ===========================================================================
def test_detector_finds_planted_peak():
    prof = np.zeros(33); prof[20] = 5.0
    det = CR.detect_from_profile(prof, "test", "test")
    assert det.critical_layer == 20
    assert det.transition_shape == "sharp"


def test_detector_calls_flat_profile_flat():
    prof = np.ones(33) + 1e-6 * np.arange(33)
    det = CR.detect_from_profile(prof, "test", "test")
    assert det.transition_shape == "flat"


def test_detector_excludes_boundary_layers():
    # A spike at the final index is usually a measurement artefact (that is
    # where the unembedding is applied), so it must not be reported.
    prof = np.zeros(20); prof[-1] = 10.0
    det = CR.detect_from_profile(prof, "test", "test", exclude_boundary=1)
    assert det.critical_layer != 19


def test_consensus_refuses_when_detectors_disagree():
    spread = {m: CR.DetectorResult(m, "k", layer, layer / 32, None, None,
                                   1.0, 0.5, 1.0, 5.0, "sharp", "ok")
              for m, layer in zip(CR.DETECTOR_SPECS, [2, 9, 16, 23, 30, 5, 27, 12])}
    cons = CR.consensus(spread, 33, min_methods_agreeing=3)
    assert cons.critical_layer_consensus is None
    assert "insufficient_agreement" in cons.consensus_status


def test_consensus_emitted_when_detectors_agree():
    agree = {m: CR.DetectorResult(m, "k", layer, layer / 32, None, None,
                                  1.0, 0.5, 1.0, 5.0, "sharp", "ok")
             for m, layer in zip(CR.DETECTOR_SPECS, [16, 17, 16, 15, 16, 17, 16, 15])}
    cons = CR.consensus(agree, 33, min_methods_agreeing=3)
    assert cons.critical_layer_consensus == 16
    assert cons.consensus_status == "ok"
    assert cons.n_methods_agreeing >= 3


def test_transition_strength_bounds():
    concentrated = np.zeros(20); concentrated[10] = 1.0
    assert abs(CR.transition_strength({"order_margin_delta": concentrated}) - 1.0) < TOL
    uniform = np.ones(20)
    assert abs(CR.transition_strength({"order_margin_delta": uniform}) - 1 / 20) < TOL


# ===========================================================================
# Storage: atomicity, checksums, recovery
# ===========================================================================
def test_atomic_write_preserves_previous_on_failure():
    tmp = Path(tempfile.mkdtemp())
    try:
        target = tmp / "result.json"
        STO.save_json(target, {"version": 1})
        try:
            with STO.atomic_path(target) as p:
                p.write_text("partial")
                raise RuntimeError("simulated crash mid-write")
        except RuntimeError:
            pass
        # The good file must survive, and no .tmp debris may be left behind.
        assert STO.load_json(target) == {"version": 1}
        assert not (tmp / "result.json.tmp").exists()
    finally:
        shutil.rmtree(tmp)


def test_manifest_skip_resume_recompute():
    tmp = Path(tempfile.mkdtemp())
    try:
        m = STO.Manifest(tmp / "manifest.jsonl")
        assert m.check("s1", "analysis") == STO.CHECK_RESUME

        out = tmp / "s1.json"
        STO.save_json(out, {"ok": True})
        m.record(STO.ManifestRecord("s1", "analysis", STO.STATUS_COMPLETE,
                                    0.0, output_path=str(out),
                                    checksum=STO.file_checksum(out)))
        assert m.check("s1", "analysis") == STO.CHECK_SKIP

        # Corrupt the file: the checksum no longer matches, so the work must
        # be redone rather than trusted.
        out.write_text('{"ok": false}')
        assert m.check("s1", "analysis") == STO.CHECK_RECOMPUTE

        out.unlink()
        assert m.check("s1", "analysis") == STO.CHECK_RECOMPUTE
    finally:
        shutil.rmtree(tmp)


def test_manifest_survives_torn_final_line():
    tmp = Path(tempfile.mkdtemp())
    try:
        path = tmp / "manifest.jsonl"
        STO.append_jsonl(path, {"sample_id": "a", "phase": "p", "status": "complete"})
        with open(path, "a") as fh:
            fh.write('{"sample_id": "b", "phase"')   # killed mid-write
        records = STO.read_jsonl(path)
        assert len(records) == 1 and records[0]["sample_id"] == "a"
    finally:
        shutil.rmtree(tmp)


def test_shard_roundtrip_and_resume_index():
    tmp = Path(tempfile.mkdtemp())
    try:
        w = STO.ShardWriter(tmp, "hidden", shard_size=2)
        rng = np.random.default_rng(0)
        data = {f"s{i}": rng.normal(size=(5, 3, 8)).astype(np.float16)
                for i in range(5)}
        for sid, arr in data.items():
            w.add(sid, {"hidden": arr})
        w.flush()

        r = STO.ShardReader(tmp, "hidden")
        assert set(r.sample_ids) == set(data)
        assert np.allclose(r.get("s3")["hidden"], data["s3"])

        # A fresh writer must see the committed IDs so a resumed run does not
        # duplicate work already on disk.
        w2 = STO.ShardWriter(tmp, "hidden", shard_size=2)
        assert w2.existing_sample_ids() == set(data)

        manifest = STO.load_json(tmp / "manifest.json")
        assert manifest["n_shards"] == 3
        for shard in manifest["shards"]:
            assert STO.file_checksum(shard["path"]) == shard["checksum"]
    finally:
        shutil.rmtree(tmp)


def test_storage_estimator_blocks_impossible_run():
    tmp = Path(tempfile.mkdtemp())
    try:
        est = STO.estimate_storage(
            n_samples=1_000_000, n_layers=80, hidden_size=8192, n_positions=64,
            n_heads=64, seq_len=2048, top_k=20,
            flags={"save_hidden_states": True, "save_attention_summaries": True,
                   "save_full_attention": False, "save_full_vocab_logits": False},
            shard_size=16, output_root=tmp, min_free_gb=1.0, vocab_size=128000)
        assert est.sufficient is False
        assert est.estimated_gb > 100
    finally:
        shutil.rmtree(tmp)


# ===========================================================================
# Datasets
# ===========================================================================
def test_synthetic_generation_is_deterministic():
    a = generate_synthetic(20, seed=42)
    b = generate_synthetic(20, seed=42)
    assert [s.sample_id for s in a] == [s.sample_id for s in b]
    assert [s.question for s in a] == [s.question for s in b]
    c = generate_synthetic(20, seed=43)
    assert [s.sample_id for s in a] != [s.sample_id for s in c]


def test_synthetic_prefix_stability():
    # Growing n must not perturb the examples already generated, so a pilot
    # subset is a strict prefix of the full set.
    small = generate_synthetic(10, seed=7)
    large = generate_synthetic(40, seed=7)
    assert [s.sample_id for s in small] == [s.sample_id for s in large[:10]]


def test_synthetic_covers_every_category_and_marks_ambiguous():
    samples = generate_synthetic(70, seed=1)
    subsets = {s.subset for s in samples}
    from nl2ms.datasets_build import SYNTHETIC_CATEGORIES
    assert subsets == set(SYNTHETIC_CATEGORIES)

    ambiguous = [s for s in samples if s.subset == "ambiguous"]
    assert ambiguous
    for s in ambiguous:
        # Deliberately underdetermined items must have no ground truth and no
        # candidate marked correct, so they can never enter an accuracy count.
        assert s.answer_spec_type == ANSWER_UNDEFINED
        assert s.ground_truth is None
        assert not any(c.is_correct for c in s.candidates)


def test_every_defined_sample_has_exactly_one_correct_candidate():
    for s in generate_synthetic(70, seed=2):
        if s.answer_spec_type == ANSWER_CLOSED_SET:
            assert sum(c.is_correct for c in s.candidates) == 1
            correct = next(c for c in s.candidates if c.is_correct)
            assert s.ground_truth == correct.label


def test_sample_ids_are_content_addressed_and_unique():
    samples = generate_synthetic(120, seed=3)
    ids = [s.sample_id for s in samples]
    assert len(set(ids)) == len(ids)
    # Identity must follow content, not row position.
    a = make_sample_id("d", "s", "question text", "A", 0)
    b = make_sample_id("d", "s", "question text", "A", 0)
    c = make_sample_id("d", "s", "different text", "A", 0)
    assert a == b and a != c


def test_sample_roundtrip():
    tmp = Path(tempfile.mkdtemp())
    try:
        samples = generate_synthetic(15, seed=5)
        path = save_samples(samples, tmp / "samples.jsonl")
        loaded = load_samples(path)
        assert len(loaded) == len(samples)
        assert loaded[0].sample_id == samples[0].sample_id
        assert loaded[0].candidates[0].label == samples[0].candidates[0].label
    finally:
        shutil.rmtree(tmp)


def test_stratified_subset_spreads_across_datasets():
    samples = generate_synthetic(60, seed=6)
    for s in samples[:20]:
        s.dataset = "other"
    sub = stratified_subset(samples, 10, seed=0)
    assert len(sub) == 10
    assert len({s.dataset for s in sub}) == 2


# ===========================================================================
# Config
# ===========================================================================
def test_config_roundtrip_and_hash_stability():
    tmp = Path(tempfile.mkdtemp())
    try:
        cfg = ExperimentConfig()
        h = cfg.config_hash()
        path = cfg.save(tmp / "config.json")
        reloaded = ExperimentConfig.load(path)
        assert reloaded.config_hash() == h
        assert reloaded.model.name == cfg.model.name
        assert reloaded.extraction.shard_size == cfg.extraction.shard_size
    finally:
        shutil.rmtree(tmp)


def test_runtime_settings_do_not_change_config_hash():
    a = ExperimentConfig()
    b = ExperimentConfig()
    b.runtime.max_runtime_hours = 3.0
    b.notes = "different note"
    # Changing the wall-clock budget must not invalidate existing checkpoints.
    assert a.config_hash() == b.config_hash()


def test_measurement_settings_do_change_config_hash():
    a = ExperimentConfig()
    b = ExperimentConfig()
    b.extraction.logit_lens_top_k = 50
    assert a.config_hash() != b.config_hash()


def test_example_config_file_loads():
    root = Path(__file__).resolve().parent.parent
    path = root / "configs" / "example_config.json"
    cfg = ExperimentConfig.load(path)
    # Hand-written config files carry a "_comment" block; it is metadata and
    # must not be mistaken for a setting.
    assert cfg.model.attn_implementation == "eager"   # required for attention
    assert cfg.storage_level == 3
    assert cfg.total_requested_samples() == 600
    assert ExperimentConfig.from_dict(cfg.to_dict()).config_hash() == cfg.config_hash()


def test_infinite_sharpness_is_sharp_not_flat():
    # A profile that is exactly zero everywhere except one layer has infinite
    # sharpness ratio. That is the sharpest possible transition, and must not
    # fall through a finiteness check into "flat".
    assert CR.classify_shape(1.0, float("inf"), 33) == "sharp"
    assert CR.classify_shape(20.0, float("inf"), 33) == "distributed"
    assert CR.classify_shape(1.0, float("nan"), 33) == "flat"


def test_storage_level_flags_are_monotone():
    prev = None
    for level in range(5):
        cfg = ExperimentConfig()
        cfg.storage_level = level
        flags = cfg.effective_flags()
        if prev is not None:
            for key in flags:
                assert flags[key] >= prev[key], f"{key} regressed at level {level}"
        prev = flags


# ===========================================================================
# Runtime control
# ===========================================================================
def test_runtime_controller_refuses_work_it_cannot_finish():
    from nl2ms.runtime import RuntimeController
    rc = RuntimeController(max_runtime_hours=1.0, reserve_minutes=10.0)
    # With no measurements yet, work must be allowed or the first shard would
    # deadlock.
    assert rc.can_afford("analysis", 10) is True
    for _ in range(5):
        rc.record("analysis", 600.0)       # 10 minutes per sample
    assert rc.can_afford("analysis", 1) is True
    assert rc.can_afford("analysis", 100) is False
    assert rc.affordable_units("analysis") is not None


def test_throughput_skips_warmup():
    from nl2ms.runtime import ThroughputTracker
    t = ThroughputTracker(warmup_skip=2)
    t.record(100.0); t.record(100.0)        # slow warm-up
    for _ in range(5):
        t.record(1.0)
    assert t.seconds_per_unit() < 2.0       # warm-up excluded


def test_heartbeat_written_and_readable():
    from nl2ms.runtime import Heartbeat, RuntimeController
    tmp = Path(tempfile.mkdtemp())
    try:
        rc = RuntimeController(1.0)
        hb = Heartbeat(tmp / "heartbeat.json", interval=0.0, controller=rc,
                       experiment_root=tmp)
        hb.beat(force=True, current_phase="analysis", current_sample="s1",
                completed_samples=3, remaining_samples=7)
        state = hb.read()
        assert state["current_phase"] == "analysis"
        assert state["completed_samples"] == 3
        assert "gpu_memory" in state and "disk" in state
    finally:
        shutil.rmtree(tmp)


# ===========================================================================
# Report integrity helpers
# ===========================================================================
def test_evidence_against_fires_on_negative_results():
    from nl2ms.report import collect_evidence_against
    results = {
        "transitions": {"shape_counts": {"sharp": 2, "distributed": 98},
                        "consensus": {"fraction_with_consensus": 0.2,
                                      "std_normalised": 0.35}},
        "causal": {"status": "ok", "cohens_d_critical_vs_random": 0.02},
        "null_models": {"layer_shuffle_jsd": {"exceeds_null": False,
                                              "p_value": 0.4}},
        "correct_vs_incorrect": [{"significant": False}, {"significant": False}],
        "no_norm_control": {"mean_correlation": 0.2},
        "confounds": {"adjusted": {"effect_attenuation": 0.8}},
        "n_models": 1,
        "accounting": {"n_completed": 20},
    }
    against = collect_evidence_against(results)
    joined = " ".join(against).lower()
    assert len(against) >= 6
    assert "no more effective than" in joined       # causal null result
    assert "does not exceed" in joined              # null-model failure
    assert "only one model" in joined


def test_evidence_against_is_quiet_on_strong_results():
    from nl2ms.report import collect_evidence_against
    results = {
        "transitions": {"shape_counts": {"sharp": 90, "distributed": 10},
                        "consensus": {"fraction_with_consensus": 0.9,
                                      "std_normalised": 0.05}},
        "causal": {"status": "ok", "cohens_d_critical_vs_random": 1.4},
        "null_models": {"layer_shuffle_jsd": {"exceeds_null": True}},
        "correct_vs_incorrect": [{"significant": True}],
        "no_norm_control": {"mean_correlation": 0.95},
        "confounds": {"adjusted": {"effect_attenuation": 0.05}},
        "n_models": 2,
        "accounting": {"n_completed": 500},
    }
    against = collect_evidence_against(results)
    # Even then, single-model and small-sample caveats must not fire falsely.
    assert not any("only one model" in a.lower() for a in against)


def test_final_report_never_asserts_the_hypothesis():
    source = (Path(__file__).resolve().parent.parent /
              "nl2ms" / "report.py").read_text().lower()
    banned = ["a phase transition was observed",
              "the hypothesis is confirmed",
              "proves that", "demonstrates that reasoning",
              "aha moment"]
    for phrase in banned:
        assert phrase not in source, f"report generator may emit: {phrase!r}"


if __name__ == "__main__":
    import traceback
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print(f"  PASS  {name}")
        except Exception as exc:
            failed += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed, {len(tests)} total")
    sys.exit(1 if failed else 0)
