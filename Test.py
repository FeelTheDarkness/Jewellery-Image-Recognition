import pandas as pd
import ast


def load_diversity_scores(div_filepath):
    """Load diversity scores and split Tag into Item1 and Item2."""
    df = pd.read_csv(div_filepath)
    if "Tag" in df.columns:

        def parse_pair(tag_str):
            try:
                items = ast.literal_eval(tag_str)
                return pd.Series(items)
            except:
                return pd.Series([None, None])

        df[["Item1", "Item2"]] = df["Tag"].apply(parse_pair)
        df = df[["Item1", "Item2", "DiversityScore"]]
    return df
