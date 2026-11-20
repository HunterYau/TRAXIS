# Envelope Pearson correlation

Each automatic experimental comparison now saves `fast_envelope_pearson_r`
(S0-related top+bottom trace) and `slow_envelope_pearson_r` (A0-related
top-bottom trace). These are Pearson correlations of the Hilbert envelopes
over the full shared 0-349.8 us interval, using the existing 5 MHz interpolation,
first-20-us baseline removal and 8192-point Hilbert padding. No fitted shift,
polarity change or time stretch is applied. This metric is independent of the
adaptive arrival gate and retains reflections and residual overlapping modes.

The range is -1 to +1; +1 is the best envelope agreement. It ignores overall
strength and does not establish accurate voltage, individual wiggles, pure
mode identity or overall physical accuracy. Zero/constant envelopes produce
JSON null and print as undefined instead of claiming agreement.

The metric prints for each mode in the console and text summaries. It is saved
in comparison.json, the comparison in result.json, all-results.json,
summaries/<frequency>khz.txt and all-results.txt. Signed individual-wiggle
correlation and envelope-shape difference are also retained.
