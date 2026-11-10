"""File-based result presentation, independent of the host HTML iframe renderer."""
import html
import math
from pathlib import Path
from urllib.parse import quote
from path_policy import inside


def cell(value):
    if value is None or isinstance(value, float) and not math.isfinite(value):
        return '—'
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        return format(value, '.8g')
    return html.escape(str(value)).replace('\\', '\\\\').replace('|', '\\|').replace('[', '\\[').replace(']', '\\]').replace('`', '\\`').replace('\n', ' ').replace('\r', ' ')


def build_result_view(run_dir, plan, state, summary, write_report):
    run_dir = Path(run_dir).resolve()
    def admitted(path):
        return path.is_file() and inside(run_dir, path.resolve())

    roots = [p for p in sorted(run_dir.iterdir()) if admitted(p) and p.suffix.lower() in ('.csv', '.html', '.json', '.joblib', '.md') and p.name != 'state.json']
    plots = [p for p in sorted((run_dir / 'plots').glob('*.png')) if admitted(p)]
    roots, root_count = roots[:128], len(roots)
    plots, plot_count = plots[:64], len(plots)
    task = plan['settings']['task']
    columns = [('Selection_Rank', 'Rank'), ('Model', 'Model')]
    if task == 'classification':
        columns += [(k, k) for k in ('CV_Accuracy', 'Val_Accuracy', 'CV_Kappa', 'Val_Kappa')]
    else:
        columns += [(k, k) for k in ('CV_RMSE', 'Val_RMSE', 'CV_R2', 'Val_R2')]
    rows = summary.get('top_pipelines', [])
    lines = ['# GB analysis results', '',
             f"Task: **{cell(task)}** · Status: **{cell(state['status'])}**", '',
             f"Samples: {cell(plan['data']['samples'])} · Features: {cell(plan['data']['features'])}",
             f"Completed candidates: {cell(state.get('completed'))}/{cell(state.get('total'))}",
             f"Successful: {cell(summary.get('successful_pipelines'))} · Failed: {cell(summary.get('failed_pipelines'))} · Holdout failures: {cell(summary.get('holdout_failed_pipelines'))}", '',
             '## Candidate evidence', '',
             f"Showing {len(rows)} of {cell(summary.get('eligible_pipeline_rows'))} eligible candidates.", '',
             '| ' + ' | '.join(title for _, title in columns) + ' |',
             '| ' + ' | '.join('---' for _ in columns) + ' |']
    lines += ['| ' + ' | '.join(cell(row.get(key)) for key, _ in columns) + ' |' for row in rows]
    if not rows:
        lines += ['', 'No eligible candidate is available. Inspect the failure evidence in the result table.']
    lines += ['', '## Interpretation', '',
              '- CV_* orders the declared candidate library; it is not an unbiased evaluation of whole-library selection.',
              '- Val_* is same-source internal holdout evidence, never used for the default ranking.',
              '- Cal_* is apparent full-calibration performance. A first-ranked candidate is not proof of universal superiority.',
              '- Execution success and predictive quality are separate. These outputs do not establish scientific validity.', '',
              '## Files', '', '[Open the result table](leaderboard_v3.csv)', '',
              'HTML is available as a supplementary file; Markdown, CSV and PNG are the default view.', '']
    for plot in plots:
        relative = plot.relative_to(run_dir).as_posix()
        lines += [f"![{cell(plot.stem)}]({quote(relative)})", '']
    report = run_dir / 'gb_summary.md'
    if report.is_symlink():
        raise ValueError('Summary destination must not be a symbolic link.')
    write_report(report, '\n'.join(lines))
    artifacts = {p.relative_to(run_dir).as_posix(): str(p) for p in [*roots, *plots] if p != report}
    artifacts[report.name] = str(report)
    default = [dict(path=str(report), role='summary', title='GB result summary'),
               dict(path=str(run_dir / 'leaderboard_v3.csv'), role='table', title='GB result table')]
    diagnostic = next((p for p in plots if p.name == f'{task}_cv_diagnostics.png'), plots[0] if plots else None)
    if diagnostic:
        default.append(dict(path=str(diagnostic), role='plot', title='GB diagnostic plot'))
    supplementary = [dict(path=str(p), role='html', title='Supplementary HTML leaderboard') for p in roots if p.suffix.lower() == '.html']
    return dict(artifacts=artifacts, presentation=dict(
        default_files=default, supplementary_files=supplementary,
        instructions='Use the native present tool with these exact path values. Default to summary, table and plot. No shell, glob or handwritten Windows-path Markdown links are needed.'),
        artifact_index=dict(root_files_found=root_count, plots_found=plot_count,
                            truncated=root_count > 128 or plot_count > 64))
