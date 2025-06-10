import json
import os

def test_data_science_article():
    with open("Data/jsons/article.json") as f:
        data = json.load(f)
    
    expected_headings = {
        "Foundations",
        "Ethical consideration in data science",
        "Cloud computing for data science"
    }
    found_headings = {section["heading"] for section in data["article_text"]}
    assert expected_headings.issubset(found_headings), "Missing expected section headings"

    foundations = next(s for s in data["article_text"] if s["heading"] == "Foundations")
    assert any("interdisciplinary field" in p for p in foundations["paragraphs"]), "Missing key phrase in Foundations"
    
    see_also_links = {link[0] for link in data["see_also_link"]}
    required_links = {"Python (programming language)", "Machine learning", "Big data"}
    assert required_links.issubset(see_also_links), "Missing required See Also links"

    for section in data["article_text"]:
        assert len(section["paragraphs"]) >= 1, f"Empty section: {section['heading']}"
        assert all(len(p) > 50 for p in section["paragraphs"]), f"Short paragraph in {section['heading']}"

if __name__ == "__main__":
    try:
        test_data_science_article()
        print("Test passed")
    except AssertionError as e:
        print("Test failed")
        raise
