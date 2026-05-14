"""DCI — Direct Corpus Interaction.

CorpusAnalyst (Section 9) and CorpusExplorer (Section 10) are the two DCI
surfaces. Analyst operates on whole documents; Explorer covers navigation,
clustering, and corpus-wide analytics.
"""

from arg.dci.analyst import CorpusAnalyst
from arg.dci.explorer import CorpusExplorer

__all__ = ["CorpusAnalyst", "CorpusExplorer"]
