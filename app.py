import io
import os
import tempfile
import traceback
from collections import defaultdict

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OPS_REQUIRED = {"JobID", "Operation", "Sequence", "Machine", "ProcessingTime"}
MAINT_REQUIRED = {"Machine", "Start", "End"}
TOL = 1e-6


def load_default_data():
    ops = pd.DataFrame(
        [
            ["J1", "Cut", 1, "M1", 20], ["J1", "Cut", 1, "M2", 24],
            ["J1", "Drill", 2, "M2", 15], ["J1", "Drill", 2, "M3", 18],
            ["J1", "Finish", 3, "M1", 12], ["J1", "Finish", 3, "M3", 15],
            ["J2", "Cut", 1, "M1", 18], ["J2", "Cut", 1, "M2", 20],
            ["J2", "Drill", 2, "M2", 22], ["J2", "Drill", 2, "M3", 20],
            ["J3", "Cut", 1, "M2", 14], ["J3", "Cut", 1, "M3", 16],
            ["J3", "Drill", 2, "M1", 16], ["J3", "Drill", 2, "M2", 18],
            ["J3", "Finish", 3, "M2", 11], ["J3", "Finish", 3, "M3", 12],
        ],
        columns=["JobID", "Operation", "Sequence", "Machine", "ProcessingTime"],
    )
    maint = pd.DataFrame(
        [["M1", 40, 55], ["M2", 35, 45], ["M3", 70, 80]],
        columns=["Machine", "Start", "End"],
    )
    return ops, maint


def find_sheet(sheet_names, keywords):
    for name in sheet_names:
        cleaned = str(name).strip().lower()
        if any(keyword in cleaned for keyword in keywords):
            return name
    return None


def read_workbook(file_path):
    workbook = pd.ExcelFile(file_path)
    ops_sheet = find_sheet(workbook.sheet_names, ["operation", "job", "routing"])
    maint_sheet = find_sheet(workbook.sheet_names, ["maintenance", "downtime", "repair"])
    if ops_sheet is None or maint_sheet is None or ops_sheet == maint_sheet:
        raise ValueError(
            "Could not find separate Operations and Maintenance sheets. "
            "Please name them 'Operations' and 'Maintenance'."
        )
    operations = pd.read_excel(workbook, sheet_name=ops_sheet)
    maintenance = pd.read_excel(workbook, sheet_name=maint_sheet)
    return operations, maintenance, ops_sheet, maint_sheet


def prepare_problem(ops_df, maint_df):
    ops = ops_df.copy()
    maint = maint_df.copy()
    ops.columns = ops.columns.astype(str).str.strip()
    maint.columns = maint.columns.astype(str).str.strip()

    missing_ops = OPS_REQUIRED - set(ops.columns)
    missing_maint = MAINT_REQUIRED - set(maint.columns)
    if missing_ops:
        raise ValueError(f"Operations sheet is missing columns: {sorted(missing_ops)}")
    if missing_maint:
        raise ValueError(f"Maintenance sheet is missing columns: {sorted(missing_maint)}")
    if ops.empty:
        raise ValueError("Operations sheet has no operation rows.")
    if ops[list(OPS_REQUIRED)].isna().any().any():
        raise ValueError("Operations sheet has blank cells in required columns.")
    if maint[list(MAINT_REQUIRED)].isna().any().any():
        raise ValueError("Maintenance sheet has blank cells in required columns.")

    for column in ["JobID", "Operation", "Machine"]:
        ops[column] = ops[column].astype(str).str.strip()
        if (ops[column] == "").any():
            raise ValueError(f"Operations column '{column}' contains a blank value.")

    seq_numeric = pd.to_numeric(ops["Sequence"], errors="raise")
    if not np.isfinite(seq_numeric.to_numpy(dtype=float)).all():
        raise ValueError("Sequence values must be finite integers.")
    if not np.allclose(seq_numeric, np.round(seq_numeric)):
        raise ValueError("Sequence values must be whole numbers.")
    ops["Sequence"] = seq_numeric.astype(int)
    ops["ProcessingTime"] = pd.to_numeric(ops["ProcessingTime"], errors="raise").astype(float)
    if (ops["Sequence"] < 1).any():
        raise ValueError("Sequence values must be positive integers.")
    if (ops["ProcessingTime"] <= 0).any() or not np.isfinite(ops["ProcessingTime"]).all():
        raise ValueError("ProcessingTime values must be finite and greater than zero.")
    if ops.duplicated(["JobID", "Operation", "Sequence", "Machine"]).any():
        raise ValueError("Duplicate JobID/Operation/Sequence/Machine rows were found.")

    if (ops.groupby(["JobID", "Sequence"])["Operation"].nunique() > 1).any():
        raise ValueError("Each JobID + Sequence must identify exactly one operation name.")
    unique_positions = ops[["JobID", "Sequence"]].drop_duplicates()
    if unique_positions.duplicated(["JobID", "Sequence"]).any():
        raise ValueError("Duplicate sequence positions were found within a job.")

    maint["Machine"] = maint["Machine"].astype(str).str.strip()
    if (maint["Machine"] == "").any():
        raise ValueError("Maintenance contains a blank machine name.")
    maint["Start"] = pd.to_numeric(maint["Start"], errors="raise").astype(float)
    maint["End"] = pd.to_numeric(maint["End"], errors="raise").astype(float)
    if not np.isfinite(maint[["Start", "End"]].to_numpy(dtype=float)).all():
        raise ValueError("Maintenance times must be finite numbers.")
    if (maint["Start"] < 0).any() or (maint["End"] <= maint["Start"]).any():
        raise ValueError("Maintenance needs Start >= 0 and End > Start.")

    for machine, group in maint.groupby("Machine"):
        intervals = sorted(zip(group["Start"], group["End"]))
        for previous, current in zip(intervals, intervals[1:]):
            if current[0] < previous[1]:
                raise ValueError(f"Maintenance intervals overlap on machine {machine}.")
    return ops, maint


def make_model(ops_df, maint_df):
    operations = {}
    insertion_order = []
    for _, row in ops_df.iterrows():
        key = (str(row["JobID"]), str(row["Operation"]), int(row["Sequence"]))
        if key not in operations:
            operations[key] = {
                "job": str(row["JobID"]),
                "operation": str(row["Operation"]),
                "sequence": int(row["Sequence"]),
                "alternatives": [],
            }
            insertion_order.append(key)
        operations[key]["alternatives"].append(
            {"machine": str(row["Machine"]), "duration": float(row["ProcessingTime"])}
        )

    jobs = defaultdict(list)
    for key, record in operations.items():
        jobs[record["job"]].append(key)

    predecessor = {}
    for job, keys in jobs.items():
        keys.sort(key=lambda item: operations[item]["sequence"])
        sequence_values = [operations[item]["sequence"] for item in keys]
        if len(sequence_values) != len(set(sequence_values)):
            raise ValueError(f"Job {job} has duplicate sequence positions.")
        for idx, key in enumerate(keys):
            predecessor[key] = keys[idx - 1] if idx else None

    maintenance = defaultdict(list)
    for _, row in maint_df.iterrows():
        maintenance[str(row["Machine"])].append((float(row["Start"]), float(row["End"])))
    for machine in maintenance:
        maintenance[machine].sort()

    return operations, insertion_order, predecessor, maintenance


def earliest_feasible_start(machine, release, duration, scheduled, maintenance):
    busy = list(maintenance.get(machine, []))
    busy.extend(
        (record["Start"], record["End"])
        for record in scheduled.values()
        if record["Machine"] == machine
    )
    busy.sort(key=lambda interval: (interval[0], interval[1]))
    time = float(release)
    for busy_start, busy_end in busy:
        if time + duration <= busy_start:
            break
        if time < busy_end and time + duration > busy_start:
            time = busy_end
    return time


def create_schedule(ops_df, maint_df, rule="FCFS", seed=0):
    operations, insertion_order, predecessor, maintenance = make_model(ops_df, maint_df)
    scheduled = {}
    rank = {key: idx for idx, key in enumerate(insertion_order)}
    rng = np.random.default_rng(seed)

    while len(scheduled) < len(operations):
        ready = [
            key for key in insertion_order
            if key not in scheduled
            and (predecessor[key] is None or predecessor[key] in scheduled)
        ]
        if not ready:
            raise RuntimeError("No ready operation exists; check the job sequences.")

        if rule == "SPT":
            chosen = min(
                ready,
                key=lambda key: (
                    min(option["duration"] for option in operations[key]["alternatives"]),
                    operations[key]["sequence"],
                    rank[key],
                ),
            )
        elif rule == "RANDOM":
            chosen = ready[int(rng.integers(0, len(ready)))]
        else:
            chosen = min(ready, key=lambda key: rank[key])

        operation = operations[chosen]
        pred_key = predecessor[chosen]
        release = scheduled[pred_key]["End"] if pred_key is not None else 0.0

        candidates = []
        for option in operation["alternatives"]:
            machine = option["machine"]
            duration = option["duration"]
            start = earliest_feasible_start(machine, release, duration, scheduled, maintenance)
            candidates.append((start + duration, start, duration, machine))
        finish, start, duration, machine = min(candidates)

        scheduled[chosen] = {
            "JobID": operation["job"],
            "Operation": operation["operation"],
            "Sequence": operation["sequence"],
            "Machine": machine,
            "ProcessingTime": duration,
            "Start": start,
            "End": finish,
        }

    result = pd.DataFrame(list(scheduled.values()))
    result["Duration"] = result["End"] - result["Start"]
    return result.sort_values(["Start", "Machine", "JobID", "Sequence"]).reset_index(drop=True)


def validate_schedule(schedule_df, ops_df, maint_df):
    errors = []
    id_cols = ["JobID", "Operation", "Sequence"]
    expected = set(
        map(tuple, ops_df[id_cols].drop_duplicates().astype({"Sequence": int}).values.tolist())
    )
    actual_rows = list(map(tuple, schedule_df[id_cols].astype({"Sequence": int}).values.tolist()))
    actual = set(actual_rows)
    if expected != actual:
        errors.append(f"Operation set mismatch: expected {len(expected)}, scheduled {len(actual)}.")
    if len(actual_rows) != len(actual):
        errors.append("A unique operation appears more than once in the schedule.")

    eligible = {}
    for _, row in ops_df.iterrows():
        key = (str(row["JobID"]), str(row["Operation"]), int(row["Sequence"]))
        eligible.setdefault(key, {})[str(row["Machine"])] = float(row["ProcessingTime"])

    machine_intervals = defaultdict(list)
    for _, row in schedule_df.iterrows():
        key = (str(row["JobID"]), str(row["Operation"]), int(row["Sequence"]))
        machine = str(row["Machine"])
        start = float(row["Start"])
        end = float(row["End"])
        duration = float(row["ProcessingTime"])
        if start < -TOL:
            errors.append(f"Negative start time for {key}.")
        if not np.isclose(end - start, duration, atol=TOL):
            errors.append(f"Duration mismatch for {key}.")
        if machine not in eligible.get(key, {}):
            errors.append(f"Ineligible machine {machine} assigned to {key}.")
        elif not np.isclose(duration, eligible[key][machine], atol=TOL):
            errors.append(f"Wrong processing time for {key} on {machine}.")
        machine_intervals[machine].append((start, end, key))

    # Check actual neighboring operations in sequence order (supports 10, 20, 30, etc.).
    for job, group in schedule_df.groupby("JobID"):
        ordered = group.sort_values("Sequence")
        rows = list(ordered.to_dict("records"))
        for previous, current in zip(rows, rows[1:]):
            if float(current["Start"]) < float(previous["End"]) - TOL:
                errors.append(
                    f"Precedence violation in {job}: {previous['Operation']} before {current['Operation']}."
                )

    for machine, intervals in machine_intervals.items():
        intervals.sort(key=lambda item: item[0])
        for previous, current in zip(intervals, intervals[1:]):
            if previous[1] > current[0] + TOL:
                errors.append(f"Machine overlap on {machine}.")

    for _, row in maint_df.iterrows():
        machine, m_start, m_end = str(row["Machine"]), float(row["Start"]), float(row["End"])
        for start, end, key in machine_intervals.get(machine, []):
            if max(start, m_start) < min(end, m_end) - TOL:
                errors.append(f"Maintenance overlap for {key} on {machine}.")

    return len(errors) == 0, errors


def machine_utilization(schedule, maintenance_df):
    horizon = float(schedule["End"].max()) if not schedule.empty else 0.0
    machines = sorted(set(schedule["Machine"].astype(str)) | set(maintenance_df["Machine"].astype(str)))
    rows = []
    for machine in machines:
        processing = float(schedule.loc[schedule["Machine"].astype(str) == machine, "Duration"].sum())
        maintenance_time = 0.0
        for _, row in maintenance_df[maintenance_df["Machine"].astype(str) == machine].iterrows():
            maintenance_time += max(
                0.0,
                min(float(row["End"]), horizon) - max(0.0, float(row["Start"])),
            )
        available = max(0.0, horizon - maintenance_time)
        utilization = 100 * processing / available if available > 0 else np.nan
        rows.append({
            "Machine": machine,
            "ProcessingTime_min": processing,
            "Maintenance_within_horizon_min": maintenance_time,
            "AvailableTime_min": available,
            "Utilization_pct": utilization,
        })
    return pd.DataFrame(rows)


def draw_gantt(schedule, maintenance, path):
    machines = sorted(set(schedule["Machine"].astype(str)) | set(maintenance["Machine"].astype(str)))
    jobs = sorted(schedule["JobID"].astype(str).unique())
    cmap = plt.get_cmap("tab20")
    colors = {job: cmap(i % 20) for i, job in enumerate(jobs)}
    fig, ax = plt.subplots(figsize=(12, max(3.5, 1.0 * len(machines))))
    for y, machine in enumerate(machines):
        for _, row in schedule[schedule["Machine"].astype(str) == machine].iterrows():
            left = float(row["Start"])
            width = float(row["End"] - row["Start"])
            ax.barh(y, width, left=left, height=0.55, color=colors[str(row["JobID"])], edgecolor="black")
            ax.text(left + width / 2, y, f"{row['JobID']}:{row['Operation']}",
                    ha="center", va="center", fontsize=8)
        for _, row in maintenance[maintenance["Machine"].astype(str) == machine].iterrows():
            ax.barh(
                y, float(row["End"]) - float(row["Start"]), left=float(row["Start"]),
                height=0.72, color="lightgrey", edgecolor="black", hatch="///", alpha=0.9,
            )
    ax.set_yticks(range(len(machines)))
    ax.set_yticklabels(machines)
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Machine")
    ax.set_title(f"Workshop schedule (makespan = {schedule['End'].max():.1f} min)")
    ax.grid(axis="x", linestyle=":", alpha=0.45)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def draw_comparison(comparison, path):
    plot_df = comparison.sort_values("Makespan_min", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(plot_df["Method"], plot_df["Makespan_min"], edgecolor="black")
    ax.set_xlabel("Makespan (minutes)")
    ax.set_title("Scheduling methods comparison")
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def analyze_workshop(uploaded_path, use_sample, method, trials):
    try:
        if use_sample:
            ops_raw, maint_raw = load_default_data()
            input_label = "Built-in demonstration dataset"
        else:
            if not uploaded_path:
                raise ValueError("Upload an Excel workbook or select the built-in sample data.")
            ops_raw, maint_raw, ops_sheet, maint_sheet = read_workbook(uploaded_path)
            input_label = f"Workbook sheets used: {ops_sheet} and {maint_sheet}"

        ops, maint = prepare_problem(ops_raw, maint_raw)
        fcfs = create_schedule(ops, maint, "FCFS")
        spt = create_schedule(ops, maint, "SPT")

        trials = int(trials)
        randomized_best = None
        randomized_makespan = float("inf")
        for seed in range(1, trials + 1):
            candidate = create_schedule(ops, maint, "RANDOM", seed=seed)
            makespan = float(candidate["End"].max())
            if makespan < randomized_makespan:
                randomized_makespan = makespan
                randomized_best = candidate

        random_name = f"Randomized Search ({trials} trials)"
        schedules = {"FCFS": fcfs, "SPT": spt, random_name: randomized_best}
        comparison = pd.DataFrame([
            {"Method": name, "Makespan_min": float(sched["End"].max())}
            for name, sched in schedules.items()
        ]).sort_values("Makespan_min").reset_index(drop=True)

        if method == "Best of all methods":
            selected_name = str(comparison.iloc[0]["Method"])
            schedule = schedules[selected_name]
        elif method == "Randomized Search":
            selected_name = random_name
            schedule = randomized_best
        else:
            selected_name = method
            schedule = schedules[method]

        valid, validation_errors = validate_schedule(schedule, ops, maint)
        utilization = machine_utilization(schedule, maint)

        workdir = tempfile.mkdtemp(prefix="workshop_dss_")
        gantt_path = os.path.join(workdir, "Workshop_Gantt.png")
        comparison_path = os.path.join(workdir, "Method_Comparison.png")
        workbook_path = os.path.join(workdir, "Workshop_Results.xlsx")
        draw_gantt(schedule, maint, gantt_path)
        draw_comparison(comparison, comparison_path)

        with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
            schedule.to_excel(writer, sheet_name="Selected Schedule", index=False)
            comparison.to_excel(writer, sheet_name="Method Comparison", index=False)
            utilization.to_excel(writer, sheet_name="Machine Utilization", index=False)
            ops.to_excel(writer, sheet_name="Input Operations", index=False)
            maint.to_excel(writer, sheet_name="Input Maintenance", index=False)
            pd.DataFrame({
                "Check": ["Feasibility validation"],
                "Result": ["PASS" if valid else "FAIL"],
                "Details": ["No violations detected" if valid else " | ".join(validation_errors)],
            }).to_excel(writer, sheet_name="Validation", index=False)

        makespan = float(schedule["End"].max())
        baseline = float(fcfs["End"].max())
        improvement = 100 * (baseline - makespan) / baseline if baseline > 0 else 0.0
        validator_line = "✅ **PASS** — no violations found." if valid else "❌ **FAIL** — see validation errors below."
        message = (
            "## Results\n"
            f"- **Data source:** {input_label}\n"
            f"- **Selected method:** {selected_name}\n"
            f"- **Makespan:** {makespan:.2f} minutes\n"
            f"- **Change vs. FCFS:** {improvement:.2f}% shorter (negative means longer)\n"
            f"- **Jobs:** {ops['JobID'].nunique()} | **Unique operations:** {ops[['JobID','Operation','Sequence']].drop_duplicates().shape[0]} | "
            f"**Machines represented:** {ops['Machine'].nunique()}\n"
            f"- **Independent validator:** {validator_line}\n\n"
            "The methods are heuristics; the result is not guaranteed to be globally optimal."
        )
        if validation_errors:
            message += "\n\n### Validation details\n" + "\n".join(f"- {e}" for e in validation_errors)

        # Make the comparison table and selected schedule easy to read in the interface.
        schedule_display = schedule[[
            "JobID", "Operation", "Sequence", "Machine", "ProcessingTime", "Start", "End", "Duration"
        ]].round(3)
        comparison_display = comparison.round(3)
        utilization_display = utilization.round(2)
        return (
            message,
            schedule_display,
            comparison_display,
            utilization_display,
            gantt_path,
            comparison_path,
            workbook_path,
        )
    except Exception as exc:
        detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        message = (
            "## Could not run the analysis\n"
            f"**Reason:** `{detail}`\n\n"
            "Check that the workbook has an Operations sheet with columns "
            "`JobID`, `Operation`, `Sequence`, `Machine`, `ProcessingTime` and a "
            "Maintenance sheet with columns `Machine`, `Start`, `End`. Times are in minutes."
        )
        empty = pd.DataFrame()
        return message, empty, empty, empty, None, None, None


with gr.Blocks(title="Workshop Production Analytics & Scheduling") as demo:
    gr.Markdown(
        """
        # 🏭 Workshop Production Analytics & Maintenance-Aware Scheduling
        Upload a workshop Excel workbook, compare scheduling heuristics, verify feasibility,
        and download the resulting schedule and charts.

        **Input format**
        - **Operations:** `JobID`, `Operation`, `Sequence`, `Machine`, `ProcessingTime`
        - **Maintenance:** `Machine`, `Start`, `End`
        - Times are in minutes. For alternative machines, put one row per eligible machine.
        """
    )
    with gr.Row():
        uploaded = gr.File(label="Upload workshop Excel workbook (.xlsx)", file_types=[".xlsx"], type="filepath")
        use_sample = gr.Checkbox(label="Use built-in sample data", value=True)
    with gr.Row():
        method = gr.Dropdown(
            choices=["Best of all methods", "FCFS", "SPT", "Randomized Search"],
            value="Best of all methods",
            label="Schedule to display",
        )
        trials = gr.Slider(10, 300, value=100, step=10, label="Randomized-search trials")
    run_btn = gr.Button("Run workshop analysis", variant="primary")
    summary = gr.Markdown()
    with gr.Tabs():
        with gr.Tab("Selected schedule"):
            schedule_table = gr.Dataframe(label="Operation schedule", interactive=False)
        with gr.Tab("Method comparison"):
            comparison_table = gr.Dataframe(label="Makespan comparison (minutes)", interactive=False)
            comparison_image = gr.Image(label="Makespan comparison chart", type="filepath")
        with gr.Tab("Machine utilization"):
            utilization_table = gr.Dataframe(label="Machine utilization", interactive=False)
    gantt_image = gr.Image(label="Gantt chart (hatched areas = planned maintenance)", type="filepath")
    with gr.Row():
        result_file = gr.File(label="Download complete Excel results")
    run_btn.click(
        fn=analyze_workshop,
        inputs=[uploaded, use_sample, method, trials],
        outputs=[
            summary, schedule_table, comparison_table, utilization_table,
            gantt_image, comparison_image, result_file,
        ],
    )
    gr.Markdown(
        """
        ---
        **Assumptions:** processing times are inputs and are not predicted. FCFS, SPT and randomized
        search are heuristics, not proofs of global optimality. Use only non-confidential data with
        a public demo link.
        """
    )

if __name__ == "__main__":
    import os
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 10000)))
