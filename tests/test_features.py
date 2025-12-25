import pytest
from src import feature_engineering


def test_add_basic_features_empty():
    import pandas as pd
    df = feature_engineering.add_basic_features(pd.DataFrame())
    assert df.empty


def test_fe_body_part_onehot():
    import pandas as pd
    df = pd.DataFrame({'body_part': ['Right Foot', 'Left Foot', 'Head', 'Other', None, 'Unknown']})
    out = feature_engineering.fe_body_part_onehot(df)
    assert list(out['body_foot']) == [1, 1, 0, 0, 0, 0]
    assert list(out['body_head']) == [0, 0, 1, 0, 0, 0]
    assert list(out['body_other']) == [0, 0, 0, 1, 0, 1]


def test_fe_game_half():
    import pandas as pd
    df = pd.DataFrame({'minute': ['10', '45+2', '46', '90+3', None, 'invalid']})
    out = feature_engineering.fe_game_half(df)
    # first four should parse to integers (may be float dtype in Series)
    assert list(out['minute_num'][:4]) == [10, 45, 46, 90]
    assert pd.isna(out['minute_num'].iloc[4])
    assert pd.isna(out['minute_num'].iloc[5])
    assert list(out['half'][:4]) == [1, 1, 2, 2]
    assert pd.isna(out['half'].iloc[4])
    assert pd.isna(out['half'].iloc[5])


def test_calculate_shot_angle():
    # Basic sanity check: returns a positive float for a typical coordinate
    a = feature_engineering.calculate_shot_angle(0, 0, goal_center_x=105, goal_y1=36.66, goal_y2=43.34)
    assert isinstance(a, float)
    assert a > 0


def test_fe_shot_type():
    import pandas as pd
    notes = ['Free kick from 25', 'Corner taken', 'Penalty awarded', 'Own goal due to deflection', None, 'Random play']
    df = pd.DataFrame({'notes': notes})
    out = feature_engineering.fe_shot_type(df)
    assert list(out['shot_type']) == ['free_kick', 'corner', 'penalty', 'own_goal', 'open_play', 'open_play']


def test_fe_assist_type():
    import pandas as pd
    df = pd.DataFrame({
        'SCA1_event': ['Pass forward', 'Cross into box', 'Shot assist', '', None, 'Dribble past'],
        'SCA2_event': ['', '', '', '', '', '']
    })
    out = feature_engineering.fe_assist_type(df, sca_cols=['SCA1_event', 'SCA2_event'])
    assert list(out['assist_type']) == ['pass', 'cross', 'secondary_shot', 'unknown', 'unknown', 'secondary_shot']


def test_fe_big_chance():
    import pandas as pd
    df = pd.DataFrame({
        'distance': [5, 6, 7, None, 3],
        'outcome': ['', 'tap in', 'one-on-one', 'Tap', 'miss']
    })
    out = feature_engineering.fe_big_chance(df)
    assert list(out['big_chance']) == [1, 1, 1, 1, 1]


def test_add_basic_features_edge_cases():
    import pandas as pd
    # missing x/y columns
    df = pd.DataFrame({'a': [1, 2]})
    out = feature_engineering.add_basic_features(df)
    assert 'distance' not in out.columns
    assert 'angle' not in out.columns

    # zero coordinates
    df2 = pd.DataFrame({'x': [0], 'y': [0]})
    out2 = feature_engineering.add_basic_features(df2)
    assert out2['distance'].iloc[0] == 0
    assert out2['angle'].iloc[0] == 0


def test_fe_body_part_onehot_edge_cases():
    import pandas as pd
    df = pd.DataFrame({'body_part': ['HEAD', 'left foot', '', None]})
    out = feature_engineering.fe_body_part_onehot(df)
    assert list(out['body_head']) == [1, 0, 0, 0]
    assert list(out['body_foot']) == [0, 1, 0, 0]
    assert list(out['body_other']) == [0, 0, 0, 0]


def test_fe_shot_type_edge_cases():
    import pandas as pd
    notes = ['freekick from 30', 'PEN', 'something cornered', 'own team penalty', '']
    df = pd.DataFrame({'notes': notes})
    out = feature_engineering.fe_shot_type(df)
    assert out['shot_type'].iloc[0] == 'free_kick'
    assert out['shot_type'].iloc[1] == 'penalty'
    assert out['shot_type'].iloc[2] == 'corner'
    # phrase contains 'pen' so current logic labels as penalty
    assert out['shot_type'].iloc[3] == 'penalty'
    assert out['shot_type'].iloc[4] == 'open_play'


def test_fe_assist_type_edge_cases():
    import pandas as pd
    # no SCA columns provided and no matching columns -> unknown
    df = pd.DataFrame({'foo': [1, 2]})
    out = feature_engineering.fe_assist_type(df, sca_cols=None)
    assert list(out['assist_type']) == ['unknown', 'unknown']

    # numeric and mixed-case event descriptions
    df2 = pd.DataFrame({'SCA 1 event': ['PASS', 'Crossing', 'take-on', None]})
    # supply explicit sca_cols matching actual column names
    out2 = feature_engineering.fe_assist_type(df2, sca_cols=['SCA 1 event'])
    assert list(out2['assist_type']) == ['pass', 'cross', 'secondary_shot', 'unknown']


def test_fe_big_chance_edge_cases():
    import pandas as pd
    # non-numeric distance strings and uppercase keywords
    df = pd.DataFrame({
        'distance': ['5.0', 'near', '10'],
        'outcome': ['TAP', 'nothing', 'one-on-one']
    })
    out = feature_engineering.fe_big_chance(df)
    # '5.0' <=6 -> big chance, 'near' -> NaN -> not from distance, 'one-on-one' -> flagged by keyword
    assert list(out['big_chance']) == [1, 0, 1]


def test_calculate_shot_angle_invalid_inputs():
    import math
    a = feature_engineering.calculate_shot_angle(None, None)
    assert math.isnan(a)
