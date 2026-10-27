"""Educational information for each supported skin-condition class.

Content is general and medically cautious: no personalised medication advice and no
instructions to self-treat. Keys are class slugs (training folder names). The default
set matches the seven HAM10000 / ISIC diagnostic categories; classes outside this set
get a generic, safe fallback entry.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DiseaseInfo:
    slug: str
    name: str
    description: str
    common_signs: tuple[str, ...]
    risk_factors: tuple[str, ...]
    care_and_prevention: tuple[str, ...]
    seek_care_when: tuple[str, ...]


_SUN_PROTECTION = "Protect skin from the sun: broad-spectrum sunscreen, protective clothing and shade, especially around midday."

_DISEASES: dict[str, DiseaseInfo] = {
    "akiec": DiseaseInfo(
        slug="akiec",
        name="Actinic Keratosis / Intraepithelial Carcinoma",
        description=(
            "Rough, scaly patches that develop after years of sun exposure. They are considered "
            "precancerous or very early skin changes, and a clinician usually evaluates them "
            "because some can progress to squamous cell skin cancer."
        ),
        common_signs=(
            "Rough, dry or scaly patch, sometimes easier to feel than to see",
            "Pink, red or brown colouring",
            "Most common on the face, ears, scalp, lips, hands and forearms",
            "May itch, burn or feel tender",
        ),
        risk_factors=(
            "Long-term or intense sun exposure and sunburns",
            "Fair skin that burns easily",
            "Older age",
            "Weakened immune system",
        ),
        care_and_prevention=(
            _SUN_PROTECTION,
            "Avoid picking or scratching the area.",
            "Check your skin regularly and note any new or changing patches.",
            "Treatment options are chosen by a clinician after examination.",
        ),
        seek_care_when=(
            "Any scaly patch that does not go away",
            "A patch that thickens, bleeds, becomes painful or grows quickly",
        ),
    ),
    "bcc": DiseaseInfo(
        slug="bcc",
        name="Basal Cell Carcinoma",
        description=(
            "The most common type of skin cancer. It usually grows slowly and rarely spreads to "
            "other parts of the body, but it can damage surrounding skin if left untreated, so "
            "professional evaluation is important."
        ),
        common_signs=(
            "Pearly or waxy bump, sometimes with visible tiny blood vessels",
            "Flat, flesh-coloured or brown scar-like patch",
            "A sore that bleeds or scabs, heals, then returns",
            "Most often on sun-exposed areas such as the face, neck and ears",
        ),
        risk_factors=(
            "Cumulative sun exposure and indoor tanning",
            "Fair skin, light eyes or hair",
            "Previous skin cancer",
            "Older age",
        ),
        care_and_prevention=(
            _SUN_PROTECTION,
            "Do not attempt to remove or treat a suspicious lesion yourself.",
            "Have regular skin checks, especially if you have had skin cancer before.",
        ),
        seek_care_when=(
            "A sore that does not heal within a few weeks",
            "A new pearly bump or a growth that changes in size, shape or colour",
            "This result suggests arranging a dermatology appointment soon",
        ),
    ),
    "bkl": DiseaseInfo(
        slug="bkl",
        name="Benign Keratosis-like Lesion",
        description=(
            "A group of common non-cancerous skin growths, including seborrheic keratoses and "
            "sun-related age spots (solar lentigines). They are usually harmless but can sometimes "
            "resemble more serious lesions."
        ),
        common_signs=(
            "Waxy, 'stuck-on' or slightly raised growths",
            "Colour ranging from light tan to dark brown",
            "Flat brown age spots on sun-exposed skin",
            "Often multiple, on the chest, back, face or hands",
        ),
        risk_factors=(
            "Increasing age",
            "Family history",
            "Cumulative sun exposure (for age spots)",
        ),
        care_and_prevention=(
            "Usually no treatment is needed unless the lesion is irritated or bothersome.",
            "Avoid picking or scratching, which can cause bleeding or infection.",
            _SUN_PROTECTION,
        ),
        seek_care_when=(
            "A lesion changes quickly, bleeds, itches persistently or becomes irritated",
            "Many new lesions appear suddenly",
            "You are unsure whether a growth is benign",
        ),
    ),
    "df": DiseaseInfo(
        slug="df",
        name="Dermatofibroma",
        description=(
            "A common, benign firm skin nodule, most often found on the lower legs. It is "
            "typically harmless and stable over time."
        ),
        common_signs=(
            "Small, firm bump that feels like a hard lump under the skin",
            "Pink, red-brown or brown colour",
            "Often dimples inward when pinched from the sides",
            "May be mildly itchy or tender",
        ),
        risk_factors=(
            "More common in adults",
            "May follow minor skin injury such as an insect bite (cause is not fully understood)",
        ),
        care_and_prevention=(
            "Usually needs no treatment.",
            "Avoid scratching or shaving over the bump, which can irritate it.",
            "Keep an eye out for any change in appearance.",
        ),
        seek_care_when=(
            "Rapid growth, bleeding, ulceration or significant pain",
            "A change in colour or shape",
            "You are unsure of the diagnosis",
        ),
    ),
    "mel": DiseaseInfo(
        slug="mel",
        name="Melanoma",
        description=(
            "A serious form of skin cancer that develops from pigment-producing cells. It is much "
            "easier to treat when found early, so any suspicious or changing pigmented lesion "
            "should be examined by a clinician promptly."
        ),
        common_signs=(
            "Asymmetry: the two halves of the spot do not match",
            "Border: edges that are irregular, ragged or blurred",
            "Colour: several shades such as brown, black, red, white or blue",
            "Diameter: larger than about 6 mm (though smaller ones can occur)",
            "Evolving: any change in size, shape, colour, or new symptoms such as itching or bleeding",
        ),
        risk_factors=(
            "Intense sun exposure and sunburns, indoor tanning",
            "Many moles or unusual (atypical) moles",
            "Personal or family history of melanoma",
            "Fair skin; weakened immune system",
        ),
        care_and_prevention=(
            "Do not wait or attempt to treat the spot yourself.",
            _SUN_PROTECTION,
            "Examine your skin monthly and consider taking dated photos to track changes.",
        ),
        seek_care_when=(
            "Promptly — a dermatologist should assess any mole or spot that looks suspicious or is changing",
            "This result suggests arranging an evaluation soon rather than monitoring at home",
        ),
    ),
    "nv": DiseaseInfo(
        slug="nv",
        name="Melanocytic Nevus (Common Mole)",
        description=(
            "Common, usually benign growths made of pigment cells. Most people have several, and "
            "most remain stable throughout life. Monitoring for change is the main recommendation."
        ),
        common_signs=(
            "Round or oval spot with a smooth, even border",
            "Uniform tan, brown or pink colour",
            "Usually smaller than about 6 mm",
            "Stable in appearance over time",
        ),
        risk_factors=(
            "Genetic tendency to have moles",
            "Sun exposure, particularly in childhood",
            "Fair skin",
        ),
        care_and_prevention=(
            _SUN_PROTECTION,
            "Check your skin monthly using the ABCDE rules (asymmetry, border, colour, diameter, evolving).",
            "Photograph moles periodically to help notice changes.",
        ),
        seek_care_when=(
            "A mole changes in size, shape or colour, or looks different from your other moles",
            "A mole itches, bleeds or becomes painful",
            "A new mole appears in adulthood and keeps changing",
        ),
    ),
    "vasc": DiseaseInfo(
        slug="vasc",
        name="Vascular Lesion",
        description=(
            "A group of skin lesions formed by blood vessels, such as cherry angiomas, "
            "angiokeratomas and pyogenic granulomas. Most are benign, but some can bleed easily "
            "and a few may resemble other conditions."
        ),
        common_signs=(
            "Red, purple or bluish spots or bumps",
            "Bright red, dome-shaped 'cherry' spots",
            "May lighten when gently pressed",
            "Some lesions bleed easily when bumped",
        ),
        risk_factors=(
            "Increasing age",
            "Hormonal changes such as pregnancy (for some types)",
            "Minor skin injury",
        ),
        care_and_prevention=(
            "Protect the area from knocks and scratching.",
            "Do not try to remove the lesion yourself.",
            "Monitor for changes in size or colour.",
        ),
        seek_care_when=(
            "Repeated bleeding or rapid growth",
            "Pain, ulceration or an unusual colour change",
            "You are unsure what the lesion is",
        ),
    ),
}


def display_name(slug: str) -> str:
    info = _DISEASES.get(slug)
    return info.name if info else slug.replace("_", " ").replace("-", " ").title()


def get_disease_info(slug: str) -> DiseaseInfo:
    """Return info for a class slug, or a cautious generic entry for unknown classes."""
    known = _DISEASES.get(slug)
    if known:
        return known
    name = display_name(slug)
    return DiseaseInfo(
        slug=slug,
        name=name,
        description=(
            f"The model suggests an image pattern associated with the category '{name}'. "
            "Detailed educational content for this category has not been added to the application."
        ),
        common_signs=("Detailed information is not available for this category.",),
        risk_factors=("Detailed information is not available for this category.",),
        care_and_prevention=(
            "Avoid self-treatment until a clinician has examined the skin.",
            _SUN_PROTECTION,
        ),
        seek_care_when=("Please have any new, changing or concerning skin finding examined by a clinician.",),
    )


def supported_slugs() -> list[str]:
    return list(_DISEASES)
