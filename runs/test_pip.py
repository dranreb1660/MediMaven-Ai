
from zenml import pipeline, step
import pandas as pd
from typing_extensions import Annotated
@step
def produce_df() -> Annotated[pd.DataFrame, "produce_df"]:
    return pd.DataFrame({"test": [1, 2, 3]})

@step
def consume_df(df: pd.DataFrame):
    print(f"Received DataFrame with columns: {df.columns}")

@pipeline(name="test_pipeline",enable_cache=False)
def test_pipeline():
    df = produce_df()
    # print(df.shape)
    consume_df(df)

test_pipeline()