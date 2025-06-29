CHEST_XRAY_KNOWLEDGE = {
    "normal": {
        "description": "A normal chest X-ray shows clear lungs, proper heart size and shape, intact ribs, and normal diaphragm position.",
        "findings": [
            "Lung fields are clear without infiltrates",
            "Cardiac silhouette is normal in size",
            "Mediastinum is normal",
            "No pleural effusion or pneumothorax",
            "Bony structures are intact"
        ],
        "treatment": "No treatment needed for normal findings",
        "keywords": ["normal", "clear", "healthy"]
    },
    "pneumonia": {
        "description": "Pneumonia typically appears as areas of consolidation or white patches in the lung fields.",
        "findings": [
            "Focal or diffuse alveolar opacities",
            "Air bronchograms may be present",
            "Possible pleural effusion",
            "Lobar consolidation in bacterial pneumonia",
            "Interstitial patterns in viral pneumonia"
        ],
        "treatment": {
            "bacterial": [
                "Outpatient: Amoxicillin/Doxycycline (healthy) or Respiratory fluoroquinolones (comorbidities)",
                "Inpatient: Ceftriaxone + Azithromycin or Levofloxacin",
                "Severe: Piperacillin-tazobactam/Carbapenems + Fluoroquinolone"
            ],
            "viral": [
                "Influenza: Oseltamivir/Zanamivir",
                "COVID-19: Remdesivir/Paxlovid (high-risk)",
                "Supportive care: Oxygen, hydration"
            ],
            "general": [
                "Oxygen if hypoxic (SpO2 < 90%)",
                "Analgesics/antipyretics (Acetaminophen/NSAIDs)",
                "IV fluids if dehydrated",
                "Hospitalize if: Age >65, RR >30, hypotension"
            ]
        },
        "keywords": ["pneumonia", "consolidation", "opacities", "infection"]
    },
    "pneumothorax": {
        "description": "Pneumothorax appears as a dark area with no lung markings, often with a visible pleural line.",
        "findings": [
            "Visible visceral pleural line",
            "Absence of lung markings peripheral to the line",
            "Mediastinal shift in tension pneumothorax",
            "Deep sulcus sign in supine patients"
        ],
        "treatment": [
            "Small (<2cm): Observation with high-flow oxygen",
            "Symptomatic: Needle aspiration or chest tube",
            "Recurrent: Pleurodesis or surgical intervention",
            "Tension: Immediate needle decompression (2nd ICS MCL)"
        ],
        "keywords": ["pneumothorax", "collapsed lung", "air leak"]
    },
    "pleural_effusion": {
        "description": "Pleural effusion appears as blunting of costophrenic angles or dense opacity in lower lung fields.",
        "findings": [
            "Meniscus sign",
            "Homogeneous opacity with concave upper border",
            "Mediastinal shift if large (>1L)"
        ],
        "treatment": [
            "Transudative: Treat underlying cause (e.g., diuretics for CHF)",
            "Exudative: Thoracentesis + analyze fluid",
            "Infected: Chest tube + antibiotics (e.g., Vancomycin + Piperacillin-tazobactam)",
            "Malignant: Pleurodesis or indwelling catheter"
        ],
        "keywords": ["effusion", "fluid", "pleural"]
    }
}

def get_response(query, detail_level="standard"):
    """
    Returns X-ray findings and treatment options
    :param query: User's question (e.g., "pneumonia treatment")
    :param detail_level: "brief"|"standard"|"detailed"
    :return: Formatted response string
    """
    query = query.lower().strip()
    
    # Extract condition and whether treatment is requested
    condition = None
    wants_treatment = "treatment" in query
    
    # Find best matching condition
    for cond, data in CHEST_XRAY_KNOWLEDGE.items():
        if cond in query or any(kw in query for kw in data["keywords"]):
            condition = cond
            break
    
    if not condition:
        available = ", ".join(CHEST_XRAY_KNOWLEDGE.keys())
        return f"I can explain: {available}. Try: 'pneumonia treatment' or 'pneumothorax findings'"
    
    return _format_condition(condition, wants_treatment, detail_level)

def _format_condition(condition, include_treatment, detail_level):
    """Helper function to format the condition response"""
    data = CHEST_XRAY_KNOWLEDGE[condition]
    response = [
        f"🩺 {condition.upper()}",
        f"\n📝 {data['description']}",
        "\n🔬 KEY FINDINGS:",
        *[f" • {f}" for f in data["findings"]]
    ]
    
    if include_treatment:
        treatment = data.get("treatment")
        if not treatment:
            response.append("\n💊 TREATMENT: Not applicable")
        elif isinstance(treatment, dict):
            response.append("\n💊 TREATMENT OPTIONS:")
            for subtype, options in treatment.items():
                response.append(f"\n  {subtype.capitalize()}:")
                response.extend([f"   • {opt}" for opt in options])
        else:
            response.append("\n💊 TREATMENT:")
            response.extend([f" • {t}" for t in treatment])
    
    # Adjust detail level (moved this before the return statement)
    if detail_level == "brief":
        return "\n".join(response[:4]) + "\n[...]"
    elif detail_level == "detailed":
        return "\n".join(response)
    return "\n".join(response)  # default "standard" level

# Example usage:
if __name__ == "__main__":
    print(get_response("pneumonia treatment", detail_level="detailed"))
    print(get_response("small pneumothorax"))
    print(get_response("normal x-ray findings"))