from app.food.db_names import DB_NAMES, search_name


def test_known_names_are_changed_to_food_db_names():
    assert search_name("계란찜") == "달걀찜"
    assert search_name("계란후라이") == "달걀부침"
    assert search_name("떡국_만두국") == "떡만두국"


def test_unknown_names_stay_the_same():
    assert search_name("김치찌개") == "김치찌개"


def test_table_has_no_empty_or_same_names():
    """빈 이름이나 바꿔도 똑같은 이름은 표에 둘 이유가 없다"""
    for ai_name, db_name in DB_NAMES.items():
        assert db_name.strip()
        assert ai_name != db_name
