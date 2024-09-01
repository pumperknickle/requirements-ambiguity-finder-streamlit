import spacy_streamlit
import streamlit as st

import spacy
from spacy_streamlit import visualize_spans
from rule_based_ambiguity_finder.indefinite_article import indefinite_article_patterns
from rule_based_ambiguity_finder.negation import negation_patterns
from rule_based_ambiguity_finder.temporal_dependencies import temporal_dep_patterns
from rule_based_ambiguity_finder.passive_voice import passive_voice_patterns

DEFAULT_TEXT = """Google was founded in September 1998 by Larry Page and Sergey Brin while they were Ph.D. students at Stanford University in California. Together they own about 14 percent of its shares and control 56 percent of the stockholder voting power through supervoting stock. They incorporated Google as a California privately held company on September 4, 1998, in California. Google was then reincorporated in Delaware on October 22, 2002."""

st.title("Requirements Ambiguity Detector")
text = st.text_area("Enter requirement to analyze", DEFAULT_TEXT, height=200)

nlp = spacy.load("en_core_web_sm")
ruler = nlp.add_pipe("span_ruler")

patterns = [{"label": "Pronoun", "pattern": [{"POS": "PRON"}]},
            {"label": "Superfluous Infinitive", "pattern": [{"LOWER": "be"}, {"POS": "ADJ"}, {"LOWER": "to"}]},
            {"label": "Superfluous Infinitive", "pattern": [{"LOWER": "be"}, {"POS": "ADJ"}, {"POS": "ADP"}]},
            {"label": "Superfluous Infinitive", "pattern": [{"LOWER": "to"}, {"POS": "VERB"}]},
            {"label": "Vague Term", "pattern": [{"LOWER": "some"}]},
            {"label": "Vague Term", "pattern": [{"LOWER": "any"}]},
            {"label": "Vague Term", "pattern": [{"LOWER": "allowable"}]},
            {"label": "Vague Term", "pattern": [{"LOWER": "several"}]},
            {"label": "Vague Term", "pattern": [{"LOWER": "many"}]},
            {"label": "Vague Term", "pattern": [{"LOWER": "a"}, {"LOWER": "lot"}, {"LOWER": "of"}]},
            {"label": "Vague Term", "pattern": [{"LOWER": "a"}, {"LOWER": "few"}]},
            {"label": "Vague Term", "pattern": [{"LOWER": "almost"}, {"LOWER": "always"}]},
            {"label": "Vague Term", "pattern": [{"LOWER": "very"}, {"LOWER": "nearly"}]},
            {"label": "Vague Term", "pattern": [{"LOWER": "nearly"}]},
            {"label": "Vague Term", "pattern": [{"LOWER": "about"}]},
            {"label": "Vague Term", "pattern": [{"LOWER": "close"}, {"LOWER": "to"}]},
            {"label": "Vague Term", "pattern": [{"LOWER": "almost"}]},
            {"label": "Vague Term", "pattern": [{"LOWER": "approximate"}]},
            {"label": "Vague Term", "pattern": [{"LOWER": "ancillary"}]},
            {"label": "Vague Term", "pattern": [{"LOWER": "relevant"}]},
            {"label": "Vague Term", "pattern": [{"LOWER": "routine"}]},
            {"label": "Vague Term", "pattern": [{"LOWER": "common"}]},
            {"label": "Vague Term", "pattern": [{"LOWER": "typical"}]},
            {"label": "Vague Term", "pattern": [{"LOWER": "routine"}]},
            {"label": "Vague Term", "pattern": [{"LOWER": "generic"}]},
            {"label": "Vague Term", "pattern": [{"LOWER": "flexible"}]},
            {"label": "Vague Term", "pattern": [{"LOWER": "expandable"}]},
            {"label": "Vague Term", "pattern": [{"LOWER": "sufficient"}]},
            {"label": "Vague Term", "pattern": [{"LOWER": "adequate"}]},
            {"label": "Vague Term", "pattern": [{"LOWER": "appropriate"}]},
            {"label": "Vague Term", "pattern": [{"LOWER": "efficient"}]},
            {"label": "Vague Term", "pattern": [{"LOWER": "secure"}]},
            {"label": "Vague Term", "pattern": [{"LOWER": "effective"}]},
            {"label": "Vague Term", "pattern": [{"LOWER": "proficient"}]},
            {"label": "Vague Term", "pattern": [{"LOWER": "reasonable"}]},
            {"label": "Vague Term", "pattern": [{"LOWER": "customary"}]},
            {"label": "Vague Adverb", "pattern": [{"LOWER": "usually"}]},
            {"label": "Vague Adverb", "pattern": [{"LOWER": "approximately"}]},
            {"label": "Vague Adverb", "pattern": [{"LOWER": "sufficiently"}]},
            {"label": "Vague Adverb", "pattern": [{"LOWER": "typically"}]},
            {"label": "Vague Adverb", "pattern": [{"LOWER": "approximately"}]},
            {"label": "Vague Adverb", "pattern": [{"POS": "ADV"}]},
            {"label": "Escape Clause", "pattern": [{"LOWER": "so"}, {"LOWER": "far"}, {"LOWER": "as"}, {"LOWER": "is"}, {"LOWER": "possible"}]},
            {"label": "Escape Clause", "pattern": [{"LOWER": "as"}, {"LOWER": "possible"}]},
            {"label": "Escape Clause", "pattern": [{"LOWER": "as"}, {"LOWER": "little"}, {"LOWER": "as"}, {"LOWER": "possible"}]},
            {"label": "Escape Clause", "pattern": [{"LOWER": "where"}, {"LOWER": "possible"}]},
            {"label": "Escape Clause", "pattern": [{"LOWER": "as"}, {"LOWER": "much"}, {"LOWER": "as"}, {"LOWER": "possible"}]},
            {"label": "Escape Clause", "pattern": [{"LOWER": "if"}, {"LOWER": "it"}, {"LOWER": "should"}, {"LOWER": "prove"}, {"LOWER": "necessary"}]},
            {"label": "Escape Clause", "pattern": [{"LOWER": "to"}, {"LOWER": "the"}, {"LOWER": "extant"}, {"LOWER": "necessary"}]},
            {"label": "Escape Clause", "pattern": [{"LOWER": "as"}, {"LOWER": "appropriate"}]},
            {"label": "Escape Clause", "pattern": [{"LOWER": "as"}, {"LOWER": "required"}]},
            {"label": "Escape Clause", "pattern": [{"LOWER": "to"}, {"LOWER": "the"}, {"LOWER": "extant"}, {"LOWER": "practical"}]},
            {"label": "Escape Clause", "pattern": [{"LOWER": "if"}, {"LOWER": "practicable"}]},
            {"label": "Escape Clause", "pattern": [{"LOWER": "as"}, {"LOWER": "necessary"}]},
            {"label": "Escape Clause", "pattern": [{"LOWER": "if"}, {"LOWER": "necessary"}]},
            {"label": "Escape Clause", "pattern": [{"LOWER": "if"}, {"LOWER": "possible"}]},
            {"label": "Open Ended Clause", "pattern": [{"LOWER": "and"}, {"LOWER": "so"}, {"LOWER": "on"}]},
            {"label": "Open Ended Clause", "pattern": [{"LOWER": "including"}, {"LOWER": "but"}, {"LOWER": "not"}, {"LOWER": "limited"}, {"LOWER": "to"}]},
            {"label": "Open Ended Clause", "pattern": [{"LOWER": "etc"}]},
            {"label": "Universal Qualification", "pattern": [{"LOWER": "all"}]},
            {"label": "Universal Qualification", "pattern": [{"LOWER": "any"}]},
            {"label": "Universal Qualification", "pattern": [{"LOWER": "both"}]},
            {"label": "Unmeasurable", "pattern": [{"LOWER": "prompt"}]},
            {"label": "Unmeasurable", "pattern": [{"LOWER": "fast"}]},
            {"label": "Unmeasurable", "pattern": [{"LOWER": "quick"}]},
            {"label": "Unmeasurable", "pattern": [{"LOWER": "rapid"}]},
            {"label": "Unmeasurable", "pattern": [{"LOWER": "minimum"}]},
            {"label": "Unmeasurable", "pattern": [{"LOWER": "maximum"}]},
            {"label": "Unmeasurable", "pattern": [{"LOWER": "minimum"}]},
            {"label": "Unmeasurable", "pattern": [{"LOWER": "optimum"}]},
            {"label": "Unmeasurable", "pattern": [{"LOWER": "nominal"}]},
            {"label": "Unmeasurable", "pattern": [{"LOWER": "easy"}, {"LOWER": "to"}, {"LOWER": "use"}]},
            {"label": "Unmeasurable", "pattern": [{"LOWER": "close"}, {"LOWER": "quickly"}]},
            {"label": "Unmeasurable", "pattern": [{"LOWER": "high"}, {"LOWER": "speed"}]},
            {"label": "Unmeasurable", "pattern": [{"LOWER": "best"}, {"LOWER": "practices"}]},
            {"label": "Unmeasurable", "pattern": [{"LOWER": "user"}, {"LOWER": "friendly"}]},
            {"label": "Combinator", "pattern": [{"ORTH": "and"}]},
            {"label": "Combinator", "pattern": [{"ORTH": "or"}]},
            {"label": "Combinator", "pattern": [{"LOWER": "then"}]},
            {"label": "Combinator", "pattern": [{"LOWER": "unless"}]},
            {"label": "Combinator", "pattern": [{"LOWER": "but"}]},
            {"label": "Combinator", "pattern": [{"LOWER": "also"}]},
            {"label": "Combinator", "pattern": [{"LOWER": "whether"}]},
            {"label": "Combinator", "pattern": [{"LOWER": "meanwhile"}]},
            {"label": "Combinator", "pattern": [{"LOWER": "whereas"}]},
            {"label": "Combinator", "pattern": [{"LOWER": "otherwise"}]},
            {"label": "Combinator", "pattern": [{"LOWER": "as"}, {"LOWER": "well"}, {"LOWER": "as"}]},
            {"label": "Combinator", "pattern": [{"LOWER": "on"}, {"LOWER": "the"}, {"LOWER": "other"}, {"LOWER": "hand"}]},
            {"label": "Unachievable Absolute", "pattern": [{"LOWER": "100%"}]},
            {"label": "Unachievable Absolute", "pattern": [{"LOWER": "all"}]},
            {"label": "Unachievable Absolute", "pattern": [{"LOWER": "every"}]},
            {"label": "Unachievable Absolute", "pattern": [{"LOWER": "always"}]},
            {"label": "Unachievable Absolute", "pattern": [{"LOWER": "never"}]},
            {"label": "Purpose Phrase", "pattern": [{"LOWER": "purpose"}, {"LOWER": "of"}]},
            {"label": "Purpose Phrase", "pattern": [{"LOWER": "intent"}, {"LOWER": "of"}]},
            {"label": "Purpose Phrase", "pattern": [{"LOWER": "reason"}, {"LOWER": "for"}]}]

patterns.extend([{"label": "IndArt", "pattern": pat} for pat in indefinite_article_patterns])
patterns.extend([{"label": "Neg", "pattern": pat} for pat in negation_patterns])
patterns.extend([{"label": "TempDep", "pattern": pat} for pat in temporal_dep_patterns])
patterns.extend([{"label": "Passive", "pattern": pat} for pat in passive_voice_patterns])
ruler.add_patterns(patterns)
doc = nlp(text)

visualize_spans(
    doc, spans_key="ruler", displacy_options={"colors": {"Negation": "#09a3d5"}}
)

