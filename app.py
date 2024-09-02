from re import escape

import spacy_streamlit
import streamlit as st

import spacy
from spacy_streamlit import visualize_spans
from rule_based_ambiguity_finder.indefinite_article import indefinite_article_patterns
from rule_based_ambiguity_finder.negation import negation_patterns
from rule_based_ambiguity_finder.temporal_dependencies import temporal_dep_patterns
from rule_based_ambiguity_finder.passive_voice import passive_voice_patterns
from rule_based_ambiguity_finder.pronoun import  pronoun_patterns
from rule_based_ambiguity_finder.infinitive import infinitive_patterns
from rule_based_ambiguity_finder.vague_term import vague_term_patterns
from rule_based_ambiguity_finder.adverb import adverb_patterns
from rule_based_ambiguity_finder.escape import escape_patterns
from rule_based_ambiguity_finder.open_ended import open_ended_patterns
from rule_based_ambiguity_finder.universal_qualitifcation import universal_qualification_patterns
from rule_based_ambiguity_finder.unmeasurable import unmeasurable_patterns
from rule_based_ambiguity_finder.combinator import combinator_patterns
from rule_based_ambiguity_finder.unachievable_absolute import unachievable_absolute_patterns
from rule_based_ambiguity_finder.purpose import purpose_patterns

DEFAULT_TEXT = "The system must allow blog visitors to sign up for the newsletter by leaving their email."

st.title("Requirements Ambiguity Detector")
text = st.text_area("Enter requirement to analyze", DEFAULT_TEXT, height=200)

nlp = spacy.load("en_core_web_sm")
ruler = nlp.add_pipe("span_ruler")

patterns = []

indefinite_article_key = "IndArt"
negation_key = "Neg"
temporal_dependency_key = "TempDep"
passive_key = "Passive"
pronoun_key = "Pron"
infinitive_key = "Inf"
vague_term_key = "Vague"
adverb_key = "Adverb"
escape_clause_key = "Esc"
open_ended_key = "OpenEnded"
universal_qualification_key = "UniQual"
unmeasurable_key = "Unmeas"
combinator_key = "Cmbntr"
unachievable_absolute_key = "UnAbs"
purpose_key = "Purpose"

patterns.extend([{"label": indefinite_article_key, "pattern": pat} for pat in indefinite_article_patterns])
patterns.extend([{"label": negation_key, "pattern": pat} for pat in negation_patterns])
patterns.extend([{"label": temporal_dependency_key, "pattern": pat} for pat in temporal_dep_patterns])
patterns.extend([{"label": passive_key, "pattern": pat} for pat in passive_voice_patterns])
patterns.extend([{"label": pronoun_key, "pattern": pat} for pat in pronoun_patterns])
patterns.extend([{"label": infinitive_key, "pattern": pat} for pat in infinitive_patterns])
patterns.extend([{"label": vague_term_key, "pattern": pat} for pat in vague_term_patterns])
patterns.extend([{"label": adverb_key, "pattern": pat} for pat in adverb_patterns])
patterns.extend([{"label": escape_clause_key, "pattern": pat} for pat in escape_patterns])
patterns.extend([{"label": open_ended_key, "pattern": pat} for pat in open_ended_patterns])
patterns.extend([{"label": universal_qualification_key, "pattern": pat} for pat in universal_qualification_patterns])
patterns.extend([{"label": unmeasurable_key, "pattern": pat} for pat in unmeasurable_patterns])
patterns.extend([{"label": combinator_key, "pattern": pat} for pat in combinator_patterns])
patterns.extend([{"label": unachievable_absolute_key, "pattern": pat} for pat in unachievable_absolute_patterns])
patterns.extend([{"label": purpose_key, "pattern": pat} for pat in purpose_patterns])

ruler.add_patterns(patterns)
doc = nlp(text)

colors = {indefinite_article_key: "#DF2935",
          negation_key: "#86BA90",
          temporal_dependency_key: "#F5F3BB",
          passive_key: "#DFA06E",
          pronoun_key: "#9381FF",
          infinitive_key: "#C94E4C",
          vague_term_key: "#B37263",
          adverb_key: "#BED7A6",
          escape_clause_key: "#906448",
          open_ended_key: "#9D967A",
          universal_qualification_key: "#E2C044",
          unmeasurable_key: "#9FB1BC",
          combinator_key: "#FF7E6B",
          unachievable_absolute_key: "#FFA69E",
          purpose_key: "#A9F0D1"}

visualize_spans(
    doc,
    spans_key="ruler",
    displacy_options={"colors": colors}
)

