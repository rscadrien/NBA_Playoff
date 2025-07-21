import os
import pandas as pd
import pytest
from Data_loading_preprocessing.data_loader import load_data

def test_load_data_basic(tmp_path):
    # Create a small CSV file
    csv_path = tmp_path / "test.csv"
    csv_path.write_text("col1,col2\n1,2\n3,4\n")
    
    df = load_data(str(csv_path))
    
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["col1", "col2"]
    assert df.shape == (2, 2)
    assert df.iloc[0, 0] == 1
    assert df.iloc[1, 1] == 4

def test_load_data_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_data("non_existing_file.csv")

def test_load_data_empty_file(tmp_path):
    empty_path = tmp_path / "empty.csv"
    empty_path.write_text("")
    
    df = load_data(str(empty_path))
    
    # Pandas returns empty DataFrame with no columns
    assert isinstance(df, pd.DataFrame)
    assert df.empty

def test_load_data_invalid_format(tmp_path):
    invalid_path = tmp_path / "invalid.csv"
    # Write something that is not CSV formatted
    invalid_path.write_text("This is not CSV content")
    
    with pytest.raises(pd.errors.ParserError):
        load_data(str(invalid_path))