# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg390::unicodedata.combining+unicodedata.normalize
# name: unicodedata_primitive
# summary: Uses unicodedata.combining, unicodedata.normalize across 3 repos
# anchor_symbols: ['unicodedata.combining', 'unicodedata.normalize']
# observed in 3 repos: ['hi-primus__optimus', 'ruc-datalab__DeepAnalyze', 'sfu-db__dataprep']...

# --- from sfu-db__dataprep::dataprep/clean/clean_duplication_utils.py::normalize_non_ascii ---
def normalize_non_ascii(val: str) -> str:
    """
    Normalize extended western characters to ascii. (remove accents)
    """
    nfkd_form = normalize("NFKD", val)
    return "".join([c for c in nfkd_form if not combining(c)])

# --- from hi-primus__optimus::optimus/engines/spark/columns.py::Cols.normalize_chars._normalize_chars ---
def _normalize_chars(value):
            value = str(value)

            # first, normalize strings:
            nfkd_str = unicodedata.normalize('NFKD', value)

            # Keep chars that has no other char combined (i.e. accents chars)
            with_out_accents = u"".join([c for c in nfkd_str if not unicodedata.combining(c)])
            return with_out_accents

# --- from hi-primus__optimus::optimus/engines/spark/columns.py::Cols.normalize_chars ---
def normalize_chars(self, input_cols="*", output_cols=None):
        """
        Remove accents in specific columns
        :param input_cols: '*', list of columns names or a single column name.
        :param output_cols:
        :return:
        """

        def _normalize_chars(value):
            value = str(value)

            # first, normalize strings:
            nfkd_str = unicodedata.normalize('NFKD', value)

            # Keep chars that has no other char combined (i.e. accents chars)
            with_out_accents = u"".join([c for c in nfkd_str if not unicodedata.combining(c)])
            return with_out_accents

        df = self.apply(input_cols, _normalize_chars, str, output_cols=output_cols,
                        meta_action=Actions.REMOVE_ACCENTS.value)
        return df

# --- from ruc-datalab__DeepAnalyze::playground/TableQA/tests/eval/wikitq_eval.py::normalize_answer ---
def normalize_answer(answer):
    """
    Normalize answer for robust comparison, handling:
    - Numbers with/without commas
    - Units (km/h, pages, etc.)
    - Lists with different separators
    - Accents and special characters
    - Different date/time formats
    """
    if not answer:
        return ""

    # Convert to lowercase and strip spaces
    answer = str(answer).strip().lower()

    # Handle numerical values with commas or units
    numeric_with_units_match = re.match(
        r"^([\d,]+)\s*(days|years|pages|km\/h|mph)$", answer
    )
    if numeric_with_units_match:
        # Extract the numeric part and remove commas
        numeric_value = numeric_with_units_match.group(1).replace(",", "")
        unit = numeric_with_units_match.group(2)
        # Return standardized format
        return f"{numeric_value} {unit}"

    # Handle simple numbers with commas
    numeric_match = re.match(r"^[\d,]+$", answer)
    if numeric_match:
        return answer.replace(",", "")  # Remove commas

    # Handle time periods
    time_period_mapping = {
        "1 week": "7 days",
        "2 weeks": "14 days",
        "1 year": "12 months",
        # Add more mappings as needed
    }
    if answer in time_period_mapping:
        return time_period_mapping[answer]

    # Try to convert to float for numerical comparison
    try:
        num = float(answer.replace(",", ""))
        if num.is_integer():
            return str(int(num))
        return str(num)
    except Exception:
        # Not a number, continue with further normalization
        pass

    # Handle lists with different separators
    if "|" in answer or "," in answer:
        items = re.split(r"[|,]\s*", answer)
        # Sort the items to handle different orders
        return "|".join(sorted([item.strip() for item in items if item.strip()]))

    # Remove accents for better character matching
    import unicodedata

    answer = "".join(
        c for c in unicodedata.normalize("NFKD", answer) if not unicodedata.combining(c)
    )

    # Final cleanup - remove unnecessary characters
    answer = re.sub(r"[^\w\s]", "", answer)  # Remove punctuation
    answer = " ".join(answer.split())  # Normalize whitespace

    return answer
