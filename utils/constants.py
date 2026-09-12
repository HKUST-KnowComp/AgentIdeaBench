"""
Shared constants for SciSynthBench.
Import from here instead of redefining locally.
"""

# The five scoring dimensions used throughout Phase 3 / Phase 4
SCORING_DIMS = ["originality", "feasibility", "clarity", "impact", "specificity"]

# Aggregation weights — justified by benchmark purpose:
#   Originality (×2):   core value of scientific proposals ("is it new?")
#   Impact (×1.5):      significance of the idea ("does it matter?")
#   Feasibility (×1):   baseline requirement ("can it be done?")
#   Clarity (×0.5):     writing quality, not scientific ability
#   Specificity (×0.5): structural detail, inflated by LLM-generated text
#
# Net effect: scientific innovation (O+I) = 64%, feasibility = 18%, writing = 18%
SCORING_WEIGHTS = {
    "originality": 2.0,
    "feasibility": 1.0,
    "clarity": 0.5,
    "impact": 1.5,
    "specificity": 0.5,
}
