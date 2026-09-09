# Update dubbing timing consumers

Remove the `min_speed` argument from `TimingSynchronizer` calls. Shorter speech
keeps its natural speed. Use `max_speed` to set the preferred acceleration limit.

Remove uses of `TimingAdjustment.was_truncated`, `truncation_seconds`, and
`excessive_slowdown`. Remove uses of `TimingSummary.truncated_count`,
`max_truncation_seconds`, `excessive_slowdown_count`, and `min_speed_factor`.
Regenerate saved timing summaries from the current pipeline. Use
`excessive_speed_count` and `max_speed_factor` to select output for listening review.

These changes apply when updating to 0.61.2. See the
[current timing reference](../reference/ai/dubbing.md#timingsummary) and
[dubbing guide](dubbing.md) for the current workflow.
