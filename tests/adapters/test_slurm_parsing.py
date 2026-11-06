"""Pure parsers of adapters.slurm against real Slurm value spellings."""

from __future__ import annotations

import pytest

from hpclab.adapters.slurm import (
    FS,
    GresItem,
    classify_sbatch_failure,
    epoch_to_iso,
    format_spec,
    gpu_counts_by_type,
    gpus_from_tres,
    node_availability,
    opt,
    parse_duration,
    parse_exit_code,
    parse_gres,
    parse_key_values,
    parse_nvidia_csv,
    parse_scontrol_config,
    parse_size_bytes,
    parse_test_only,
    parse_version,
    split_job_id,
    split_records,
    split_top_level,
    text_or_none,
)


@pytest.mark.parametrize(
    "value", [None, "", "  ", "Unknown", "None", "N/A", "n/a", "(null)", "None assigned", "INVALID"]
)
def test_opt_placeholders_are_none(value: str | None) -> None:
    assert opt(value) is None


def test_opt_keeps_real_values_including_zero() -> None:
    assert opt(" gpu ") == "gpu"
    assert opt("0") == "0"


def test_text_or_none_only_drops_empty_and_null() -> None:
    assert text_or_none("None") == "None"  # a job may really be called None
    assert text_or_none("(null)") is None
    assert text_or_none("") is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1790000000", "2026-09-21T14:13:20.000Z"),
        ("Unknown", None),
        ("N/A", None),
        ("None", None),
        ("0", None),
        ("", None),
        ("-5", None),
    ],
)
def test_epoch_to_iso(raw: str, expected: str | None) -> None:
    assert epoch_to_iso(raw) == expected


def test_epoch_to_iso_rejects_formatted_times(caplog: pytest.LogCaptureFixture) -> None:
    # A formatted time means SLURM_TIME_FORMAT was ignored: the timezone is unknown, so no guess.
    assert epoch_to_iso("2026-09-21T13:46:40") is None
    assert "SLURM_TIME_FORMAT" in caplog.text


@pytest.mark.parametrize(
    ("raw", "seconds"),
    [
        ("1-02:03:04", 93784),
        ("7-00:00:00", 604800),
        ("6:00:00", 21600),
        ("12:34", 754),
        ("0:00", 0),
        ("00:01.234", 1),
        ("1-02:03:04.123", 93784),
        ("UNLIMITED", None),
        ("INVALID", None),
        ("Partition_Limit", None),
        ("infinite", None),
        ("", None),
    ],
)
def test_parse_duration(raw: str, seconds: int | None) -> None:
    assert parse_duration(raw) == seconds


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("0:0", (0, 0)),
        ("1:0", (1, 0)),
        ("0:15", (0, 15)),
        ("0:125", (0, 125)),
        ("", (None, None)),
        ("x:y", (None, None)),
    ],
)
def test_parse_exit_code(raw: str, expected: tuple[int | None, int | None]) -> None:
    assert parse_exit_code(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("123", ("123", None)), ("123_4", ("123_4", "4")), ("123_[5-9]", ("123_[5-9]", "[5-9]"))],
)
def test_split_job_id(raw: str, expected: tuple[str, str | None]) -> None:
    assert split_job_id(raw) == expected


@pytest.mark.parametrize(
    ("tres", "gpus"),
    [
        ("cpu=8,mem=64G,node=1,billing=8,gres/gpu=2,gres/gpu:h100=2", 2),
        ("billing=8,cpu=8,gres/gpu:h100=1,gres/gpu=1,mem=64G,node=1", 1),
        ("cpu=16,mem=128G,node=2,gres/gpu:h200=4", 4),
        ("cpu=16,gres/gpu:a100=2,gres/gpu:v100=1", 3),
        ("cpu=4,mem=32G,node=1,billing=4", 0),
        ("", None),
        (None, None),
    ],
)
def test_gpus_from_tres(tres: str | None, gpus: int | None) -> None:
    assert gpus_from_tres(tres) == gpus


def test_split_top_level_respects_parentheses() -> None:
    assert split_top_level("gpu:h100:3(IDX:0,2-3),shard:0") == ["gpu:h100:3(IDX:0,2-3)", "shard:0"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("gpu:h100:8(S:0-1)", [GresItem("gpu", "h100", 8, "S:0-1")]),
        ("gpu:8", [GresItem("gpu", None, 8)]),
        ("gpu", [GresItem("gpu", None, 1)]),
        ("gpu:b200:8,shard:b200:32", [GresItem("gpu", "b200", 8), GresItem("shard", "b200", 32)]),
        ("gpu:h100:3(IDX:0-2)", [GresItem("gpu", "h100", 3, "IDX:0-2")]),
        ("gpu:h100:5(IDX:0,2-5)", [GresItem("gpu", "h100", 5, "IDX:0,2-5")]),
        ("gpu:0", [GresItem("gpu", None, 0)]),
        ("gpu:(null):0(IDX:N/A)", [GresItem("gpu", None, 0, "IDX:N/A")]),
        ("gpu:a100:no_consume:4", [GresItem("gpu", "a100", 4)]),
        ("mps:400K", [GresItem("mps", None, 400 * 1024)]),
        ("(null)", []),
        ("", []),
    ],
)
def test_parse_gres(raw: str, expected: list[GresItem]) -> None:
    assert parse_gres(raw) == expected


@pytest.mark.parametrize(
    ("state", "availability"),
    [
        ("idle", "free"),
        ("mixed", "free"),
        ("allocated", "busy"),
        ("completing", "busy"),
        ("reserved", "busy"),
        ("planned", "busy"),
        ("drained", "unavailable"),
        ("draining", "unavailable"),
        ("down", "unavailable"),
        ("down*", "unavailable"),
        ("idle*", "unavailable"),
        ("idle~", "unavailable"),
        ("mixed+drain", "unavailable"),
        ("maint", "unavailable"),
        ("fail", "unavailable"),
        ("", "unavailable"),
    ],
)
def test_node_availability(state: str, availability: str) -> None:
    assert node_availability(state) == availability


def test_gpu_counts_by_type_typed_and_untyped() -> None:
    assert gpu_counts_by_type("gpu:h100:8(S:0-1)", "gpu:h100:3(IDX:0-2)") == {"h100": (8, 3)}
    # untyped "used" is attributed to the node's single type
    assert gpu_counts_by_type("gpu:h200:8", "gpu:0") == {"h200": (8, 0)}
    assert gpu_counts_by_type("gpu:8", "gpu:2(IDX:0-1)") == {"untyped": (8, 2)}
    # shards are not GPUs
    assert gpu_counts_by_type("gpu:b200:8,shard:b200:32", "gpu:b200:2(IDX:0-1),shard:b200:0") == {
        "b200": (8, 2)
    }
    assert gpu_counts_by_type("(null)", "(null)") == {}
    two = gpu_counts_by_type("gpu:a100:4(S:0),gpu:v100:2(S:1)", "gpu:a100:1(IDX:0),gpu:v100:2(IDX:4-5)")
    assert two == {"a100": (4, 1), "v100": (2, 2)}


def test_split_records_joins_continuations_of_the_last_field() -> None:
    text = f"1{FS}a{FS}first line\nsecond line\n2{FS}b{FS}ok\n\n"
    assert split_records(text, 3) == [["1", "a", "first line\nsecond line"], ["2", "b", "ok"]]


def test_split_records_last_field_may_contain_the_delimiter() -> None:
    assert split_records("a|b|c|d", 3, sep="|") == [["a", "b", "c|d"]]


def test_split_records_newline_inside_a_middle_field_keeps_every_record_intact() -> None:
    # A foreign job whose comment spans two lines must neither vanish nor leak into the
    # previous job's last field (that job may be one hpclab manages).
    text = "\n".join(
        [
            FS.join(["1", "RUNNING", "hpclab:tok", "None"]),
            FS.join(["2", "PENDING", "first line"]),
            FS.join(["second line", "Resources"]),
            FS.join(["3", "RUNNING", "(null)", "None"]),
        ]
    )
    assert split_records(text + "\n", 4) == [
        ["1", "RUNNING", "hpclab:tok", "None"],
        ["2", "PENDING", "first line\nsecond line", "Resources"],
        ["3", "RUNNING", "(null)", "None"],
    ]


def test_split_records_drops_a_truncated_final_record() -> None:
    text = FS.join(["1", "a", "ok"]) + "\n" + FS.join(["2", "b"])  # output cut mid-record
    assert split_records(text, 3) == [["1", "a", "ok"]]


def test_format_spec_uses_width_zero_and_suffix_delimiter() -> None:
    assert format_spec(["JobArrayID", "State", "Reason"]) == f"JobArrayID:0{FS},State:0{FS},Reason:0"


@pytest.mark.parametrize(
    ("stderr", "kind", "code"),
    [
        (
            "sbatch: error: invalid partition specified: gpux\n"
            "sbatch: error: Batch job submission failed: Invalid partition name specified",
            "rejected",
            "invalid_partition",
        ),
        (
            "sbatch: error: Batch job submission failed: Invalid account or account/partition combination specified",
            "rejected",
            "invalid_account",
        ),
        (
            "sbatch: error: Batch job submission failed: Requested time limit is invalid (missing or exceeds some limit)",
            "rejected",
            "invalid_time_limit",
        ),
        (
            "sbatch: error: Batch job submission failed: Requested node configuration is not available",
            "rejected",
            "node_config_unavailable",
        ),
        (
            "sbatch: error: Memory specification can not be satisfied\n"
            "sbatch: error: Batch job submission failed: Requested node configuration is not available",
            "rejected",
            "memory_unsatisfiable",
        ),
        (
            "sbatch: error: QOSMaxSubmitJobPerUserLimit\n"
            "sbatch: error: Batch job submission failed: Job violates accounting/QOS policy "
            "(job submit limit, user's size and/or time limits)",
            "rejected",
            "submit_limit",
        ),
        ("sbatch: error: AssocMaxSubmitJobLimit", "rejected", "submit_limit"),
        ("sbatch: error: Batch job submission failed: Invalid qos specification", "rejected", "invalid_qos"),
        (
            "sbatch: error: Batch job submission failed: Invalid generic resource (gres) specification",
            "rejected",
            "invalid_gres",
        ),
        ("sbatch: error: Unable to open file job.sbatch", "rejected", "script_unreadable"),
        (
            "sbatch: unrecognized option '--gpu'\nTry \"sbatch --help\" for more information",
            "rejected",
            "invalid_option",
        ),
        (
            "sbatch: error: Batch job submission failed: Job dependency problem",
            "rejected",
            "dependency_invalid",
        ),
        (
            "sbatch: error: You must specify an account\nsbatch: error: Batch job submission failed: Unspecified error",
            "rejected",
            "controller_refused",
        ),
        (
            "sbatch: error: Batch job submission failed: Socket timed out on send/recv operation",
            "ambiguous",
            "socket_timeout",
        ),
        (
            "sbatch: error: Batch job submission failed: Unable to contact slurm controller (connect failure)",
            "ambiguous",
            "controller_unreachable",
        ),
        (
            "sbatch: error: Slurm temporarily unable to accept job, sleeping and retrying\n"
            "sbatch: error: Batch job submission failed: Resource temporarily unavailable",
            "ambiguous",
            "controller_busy",
        ),
        (
            "sbatch: error: Batch job submission failed: Zero Bytes were transmitted or received",
            "ambiguous",
            "zero_bytes",
        ),
        ("", "ambiguous", "unrecognized_failure"),
        ("Segmentation fault (core dumped)", "ambiguous", "unrecognized_failure"),
    ],
)
def test_classify_sbatch_failure(stderr: str, kind: str, code: str) -> None:
    failure = classify_sbatch_failure(stderr)
    assert (failure.kind, failure.reason_code) == (kind, code)
    assert failure.message


@pytest.mark.parametrize(
    "stderr",
    [
        "env: 'sbatch': No such file or directory",  # GNU env, C locale
        "env: ‘sbatch’: No such file or directory",  # GNU env, UTF-8 locale
        "env: sbatch: No such file or directory",  # BSD env
        "bash: sbatch: command not found",
    ],
)
def test_classify_missing_sbatch_is_a_definite_refusal(stderr: str) -> None:
    # env could not exec sbatch: nothing can have been submitted. Treating this as ambiguous
    # parks every attempt in "uncertain" for the whole reconcile window on a PATH problem.
    failure = classify_sbatch_failure(stderr)
    assert (failure.kind, failure.reason_code) == ("rejected", "sbatch_unavailable")


@pytest.mark.parametrize(
    "stderr",
    [
        # sbatch parses every option (command line and #SBATCH lines) before it contacts the
        # controller and exits on the first bad one: nothing can have been queued.
        "sbatch: error: Invalid --signal specification",  # lead_s > 65535 (uint16 in Slurm)
        "sbatch: error: Invalid --mem-per-gpu specification",
        "sbatch: error: Invalid --gpus-per-task specification",
        "sbatch: error: Invalid --distribution specification",
        "sbatch: error: Invalid --open-mode argument: bogus.",
        "sbatch: error: Invalid --nice value: x",
        "sbatch: error: invalid memory constraint 64GB",  # Slurm < 20.02
        "sbatch: error: --mail-type=BOGUS invalid",
        "sbatch: option '--time' requires an argument\nTry \"sbatch --help\" for more information",
        "sbatch: option requires an argument -- 't'",
        "sbatch: option '--parsable' doesn't allow an argument",
        "sbatch: option '--me' is ambiguous; possibilities: '--mem' '--mem-per-cpu' '--mem-per-gpu'",
    ],
)
def test_classify_client_side_option_errors_are_definite_refusals(stderr: str) -> None:
    # Classified "ambiguous", a typo in extra_sbatch parks the attempt in SUBMISSION_UNCERTAIN
    # for the whole reconcile window and then blocks it on a human, and the prepare-time
    # --test-only check reports "could not be evaluated" instead of the refusal.
    failure = classify_sbatch_failure(stderr)
    assert (failure.kind, failure.reason_code) == ("rejected", "invalid_option")
    assert failure.message


def test_classify_invalid_time_keeps_its_own_code() -> None:
    failure = classify_sbatch_failure("sbatch: error: Invalid --time specification")
    assert (failure.kind, failure.reason_code) == ("rejected", "invalid_time_limit")


@pytest.mark.parametrize(
    "noise",
    [
        "Access denied to /etc/motd.d\nInvalid account settings in ~/.bashrc",
        "module: invalid option -- 'q'",
        "conda: error: Invalid --prefix specification\nmodule: option '--x' requires an argument",
    ],
)
def test_classify_ignores_refusal_words_not_written_by_sbatch(noise: str) -> None:
    # Login-shell noise on stderr must never turn an unexplained failure into "nothing queued".
    assert classify_sbatch_failure(noise + "\nSegmentation fault").kind == "ambiguous"


@pytest.mark.parametrize("reason", ["Insane message length", "Header lengths are longer than data received"])
def test_classify_corrupt_reply_is_ambiguous(reason: str) -> None:
    # A garbled *reply* can follow a job that was created.
    failure = classify_sbatch_failure(f"sbatch: error: Batch job submission failed: {reason}")
    assert failure.kind == "ambiguous"


def test_classify_prefers_final_line_over_earlier_retry_noise() -> None:
    stderr = (
        "sbatch: error: Slurm temporarily unable to accept job, sleeping and retrying\n"
        "sbatch: error: Batch job submission failed: Invalid account or account/partition combination specified"
    )
    assert classify_sbatch_failure(stderr).kind == "rejected"


def test_parse_test_only_epoch_and_iso() -> None:
    est = parse_test_only(
        "sbatch: Job 4244 to start at 1790000000 using 8 processors on nodes gpu-node-4 in partition gpu"
    )
    assert est is not None
    assert (est.job_id, est.start_time, est.processors, est.nodes, est.partition) == (
        "4244",
        "2026-09-21T14:13:20.000Z",
        8,
        "gpu-node-4",
        "gpu",
    )
    naive = parse_test_only(
        "sbatch: Job 1 to start at 2026-09-21T13:46:40 using 1 processors on nodes n1 in partition p"
    )
    assert naive is not None and naive.start_time is None  # unknown timezone -> no guess
    assert parse_test_only("sbatch: error: something") is None


def test_parse_version_and_key_values() -> None:
    assert parse_version("slurm 23.02.7") == (23, 2, 7)
    assert parse_version("slurm 25.05.0-0rc1") == (25, 5, 0)
    assert parse_version("garbage") is None
    assert parse_key_values("PartitionName=gpu Default=YES Nodes=gpu-node-[1-5]")["Nodes"] == "gpu-node-[1-5]"


def test_parse_scontrol_config() -> None:
    conf = parse_scontrol_config(
        "Configuration data as of 2026-09-27T10:00:00\n"
        "AccountingStoreFlags    = job_comment,job_script\n"
        "AccountingStorageBackupHost = (null)\n"
        "MinJobAge               = 300 sec\n"
    )
    assert conf["AccountingStoreFlags"] == "job_comment,job_script"
    assert conf["MinJobAge"] == "300 sec"
    assert conf["AccountingStorageBackupHost"] == "(null)"


@pytest.mark.parametrize(
    ("raw", "n"),
    [
        ("4194304K", 4194304 * 1024),
        ("12.5G", int(12.5 * 1024**3)),
        ("0", 0),
        ("512", 512),
        ("", None),
        ("x", None),
    ],
)
def test_parse_size_bytes(raw: str, n: int | None) -> None:
    assert parse_size_bytes(raw) == n


def test_parse_nvidia_csv_handles_not_available() -> None:
    samples = parse_nvidia_csv(
        "0, NVIDIA H100 80GB HBM3, 97, 71234, 81559, 612.34\n"
        "1, NVIDIA H100 80GB HBM3, [N/A], 0, 81559, [Not Supported]\n"
        "garbage line\n"
    )
    assert len(samples) == 2
    assert samples[0].util_percent == 97.0 and samples[0].power_w == 612.34
    assert samples[1].util_percent is None and samples[1].power_w is None and samples[1].mem_used_mib == 0.0
