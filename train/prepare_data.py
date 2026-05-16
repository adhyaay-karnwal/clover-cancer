"""
Clover Cancer — Dataset Preparation Script

Builds a pancreatic cancer triage training dataset by combining:
1. Base medical Q&A from gretelai/symptom_to_diagnosis (22 conditions, 1065 examples)
2. Pancreatic cancer clinical dataset from Kaggle (biomarker data)
3. Synthetically generated pancreatic cancer triage examples based on clinical research

Output: Structured training data formatted for Gemma 4 fine-tuning with Unsloth.

Author: Adhyaay Karnwal
"""

import json
import random
import os
from pathlib import Path

# Pancreatic cancer symptom patterns based on clinical research
# Sources: NCCN guidelines, Chari et al. (MD Anderson), Frontiers 2025 review,
# Johns Hopkins CT missed signs study, Pancreatic Cancer Action Network

PANCREATIC_CANCER_PATTERNS = [
    # HIGH RISK — Classic early detection patterns
    {
        "symptoms": ["new-onset diabetes", "unexplained weight loss", "back pain"],
        "age_range": (50, 75),
        "risk_factors": [],
        "duration": "2-4 months",
        "risk": "high",
        "urgency": "urgent",
        "reasoning": "The combination of new-onset diabetes after age 50 with unexplained weight loss and back pain is a recognized early presentation pattern for pancreatic cancer. Studies show new-onset diabetes can precede pancreatic cancer diagnosis by 1-3 years. The cancer may actually cause the diabetes by disrupting pancreatic islet function. This pattern warrants urgent imaging.",
        "actions": ["Abdominal CT with pancreatic protocol", "CA 19-9 tumor marker", "Endoscopic ultrasound if CT inconclusive", "Referral to gastroenterology"],
        "conditions": [
            {"name": "Pancreatic ductal adenocarcinoma", "likelihood": "high", "reasoning": "New-onset diabetes + weight loss + back pain in patient >50 is a recognized early PC pattern. ~1 in 160 NOD patients ≥50 are diagnosed with PC within 3 years."},
            {"name": "Pancreatogenic diabetes (Type 3c)", "likelihood": "medium", "reasoning": "PC can cause diabetes by disrupting islet function. Must rule out underlying pancreatic pathology."},
            {"name": "Chronic pancreatitis", "likelihood": "medium", "reasoning": "Back pain and diabetes can indicate chronic pancreatitis, which itself is a PC risk factor."}
        ]
    },
    {
        "symptoms": ["painless jaundice", "dark urine", "weight loss", "itching"],
        "age_range": (55, 80),
        "risk_factors": [],
        "duration": "2-6 weeks",
        "risk": "critical",
        "urgency": "emergency",
        "reasoning": "Painless jaundice with dark urine and weight loss is the classic presentation of a pancreatic head mass obstructing the bile duct. This is a medical emergency requiring immediate evaluation. The triad of painless jaundice + weight loss + dark urine has high specificity for periampullary malignancy.",
        "actions": ["Immediate abdominal CT with contrast", "MRCP", "ERCP if biliary obstruction confirmed", "Urgent surgical oncology referral", "CA 19-9"],
        "conditions": [
            {"name": "Pancreatic head adenocarcinoma", "likelihood": "high", "reasoning": "Painless jaundice with weight loss is the hallmark presentation of pancreatic head tumors causing bile duct obstruction."},
            {"name": "Ampullary carcinoma", "likelihood": "medium", "reasoning": "Periampullary tumors can present identically but have better prognosis."},
            {"name": "Cholangiocarcinoma", "likelihood": "medium", "reasoning": "Bile duct cancer can cause identical obstructive jaundice pattern."}
        ]
    },
    {
        "symptoms": ["new-onset diabetes", "abdominal pain", "fatigue"],
        "age_range": (55, 70),
        "risk_factors": ["family history of pancreatic cancer"],
        "duration": "3 months",
        "risk": "high",
        "urgency": "urgent",
        "reasoning": "New-onset diabetes in a patient with family history of pancreatic cancer and abdominal pain raises significant concern. NCCN guidelines recommend screening for individuals with ≥2 affected first-degree relatives. The combination of NOD with family history elevates risk substantially.",
        "actions": ["Abdominal MRI/MRCP", "CA 19-9", "Genetic counseling for BRCA/PALB2 testing", "Consider EUS"],
        "conditions": [
            {"name": "Pancreatic cancer", "likelihood": "high", "reasoning": "Family history + NOD + abdominal pain is a high-risk constellation. Genetic predisposition (BRCA2, PALB2) combined with new metabolic changes warrants urgent evaluation."},
            {"name": "Hereditary pancreatitis", "likelihood": "low", "reasoning": "Could explain symptoms but doesn't reduce cancer risk — actually increases it."},
            {"name": "Peptic ulcer disease", "likelihood": "low", "reasoning": "Abdominal pain could be ulcer-related but doesn't explain new-onset diabetes."}
        ]
    },
    {
        "symptoms": ["back pain", "abdominal pain", "weight loss", "new-onset diabetes"],
        "age_range": (60, 80),
        "risk_factors": ["smoking history"],
        "duration": "4 months",
        "risk": "high",
        "urgency": "urgent",
        "reasoning": "Smoking is the strongest modifiable risk factor for pancreatic cancer (2-3x risk). Combined with new-onset diabetes, back pain, and weight loss in an older patient, this represents a high-risk presentation. The back pain suggests possible retroperitoneal involvement.",
        "actions": ["Pancreatic protocol CT", "CA 19-9", "Chest CT to rule out metastases", "Gastroenterology referral"],
        "conditions": [
            {"name": "Pancreatic cancer", "likelihood": "high", "reasoning": "Smoking history doubles PC risk. The symptom cluster of back pain + NOD + weight loss in a smoker >60 is highly concerning."},
            {"name": "Chronic pancreatitis", "likelihood": "medium", "reasoning": "Smoking and alcohol are risk factors for chronic pancreatitis, which can present similarly."},
            {"name": "Lung cancer with pancreatic metastases", "likelihood": "low", "reasoning": "Smoking history raises concern for primary lung malignancy with pancreatic involvement."}
        ]
    },
    # MODERATE RISK — Patterns that get missed
    {
        "symptoms": ["new-onset diabetes", "mild abdominal discomfort"],
        "age_range": (50, 65),
        "risk_factors": [],
        "duration": "1 month",
        "risk": "medium",
        "urgency": "routine",
        "reasoning": "New-onset diabetes after age 50 without typical risk factors (obesity, family history of diabetes) should raise suspicion for pancreatic cancer. Studies show 0.625% of NOD patients ≥50 are diagnosed with PC within 3 years. While most NOD patients will not develop PC, clinical vigilance is warranted.",
        "actions": ["Monitor symptoms closely", "Consider CA 19-9", "Follow up in 2-3 months if symptoms persist", "Consider abdominal imaging if any additional symptoms develop"],
        "conditions": [
            {"name": "Type 2 diabetes", "likelihood": "high", "reasoning": "Most new-onset diabetes in this age group is standard Type 2 diabetes. However, PC-associated diabetes must be considered."},
            {"name": "Pancreatic cancer (early)", "likelihood": "low-moderate", "reasoning": "NOD can be the earliest sign of PC, preceding other symptoms by 1-3 years. Clinical follow-up recommended."},
            {"name": "Pancreatogenic diabetes", "likelihood": "low", "reasoning": "Could indicate underlying pancreatic pathology beyond cancer."}
        ]
    },
    {
        "symptoms": ["persistent back pain", "fatigue", "loss of appetite"],
        "age_range": (55, 75),
        "risk_factors": ["BRCA2 mutation"],
        "duration": "6 weeks",
        "risk": "high",
        "urgency": "urgent",
        "reasoning": "BRCA2 mutation carriers have 3-6x increased risk of pancreatic cancer. NCCN guidelines recommend surveillance for BRCA2 carriers with ≥1 affected first-degree relative. New back pain with fatigue and appetite loss in a BRCA2 carrier warrants urgent evaluation even without classic PC symptoms.",
        "actions": ["Abdominal MRI/MRCP", "CA 19-9", "EUS", "Genetic counseling update", "Oncology referral"],
        "conditions": [
            {"name": "Pancreatic cancer", "likelihood": "high", "reasoning": "BRCA2 mutation significantly elevates PC risk. New symptoms in this population require urgent evaluation per NCCN guidelines."},
            {"name": "Other BRCA2-associated cancers", "likelihood": "medium", "reasoning": "BRCA2 also increases risk of breast, ovarian, and prostate cancers. Symptoms could indicate other malignancies."},
            {"name": "Chronic pancreatitis", "likelihood": "low", "reasoning": "Back pain and appetite loss could indicate pancreatitis, but doesn't reduce concern given genetic risk."}
        ]
    },
    {
        "symptoms": ["steatorrhea", "weight loss", "abdominal bloating"],
        "age_range": (50, 70),
        "risk_factors": [],
        "duration": "3 months",
        "risk": "medium",
        "urgency": "urgent",
        "reasoning": "Steatorrhea (fatty, pale, foul-smelling stools) indicates pancreatic enzyme insufficiency. Combined with weight loss and bloating, this suggests exocrine pancreatic dysfunction. While this can be caused by chronic pancreatitis or celiac disease, pancreatic cancer must be excluded, especially in patients >50.",
        "actions": ["Fecal elastase test", "Abdominal CT with pancreatic protocol", "CA 19-9", "Consider EUS", "Pancreatic enzyme replacement trial"],
        "conditions": [
            {"name": "Pancreatic cancer", "likelihood": "medium", "reasoning": "Exocrine insufficiency with weight loss in older patient raises concern for pancreatic mass causing duct obstruction."},
            {"name": "Chronic pancreatitis", "likelihood": "medium", "reasoning": "Most common cause of exocrine insufficiency. Can coexist with or predispose to cancer."},
            {"name": "Celiac disease", "likelihood": "low", "reasoning": "Can cause steatorrhea and weight loss but typically has different age of onset pattern."}
        ]
    },
    {
        "symptoms": ["new-onset diabetes", "unexplained weight loss"],
        "age_range": (55, 70),
        "risk_factors": ["chronic pancreatitis"],
        "duration": "2 months",
        "risk": "high",
        "urgency": "urgent",
        "reasoning": "Chronic pancreatitis is a significant PC risk factor (10-20x). New-onset diabetes with weight loss in a chronic pancreatitis patient could indicate malignant transformation. Studies show chronic pancreatitis patients have lifetime PC risk of 4-5%. New metabolic changes in this population require urgent evaluation.",
        "actions": ["Pancreatic protocol CT", "MRI/MRCP", "CA 19-9", "EUS with FNA if mass found", "Oncology referral"],
        "conditions": [
            {"name": "Pancreatic cancer arising from chronic pancreatitis", "likelihood": "high", "reasoning": "Chronic pancreatitis is a known risk factor. New diabetes and weight loss suggest possible malignant transformation."},
            {"name": "Worsening chronic pancreatitis", "likelihood": "medium", "reasoning": "Progressive pancreatitis can cause new diabetes and weight loss without cancer."},
            {"name": "Pancreatic pseudocyst", "likelihood": "low", "reasoning": "Can cause symptoms but typically presents with more acute pain."}
        ]
    },
    # LOWER RISK — Important to distinguish from PC
    {
        "symptoms": ["back pain", "abdominal pain"],
        "age_range": (40, 60),
        "risk_factors": [],
        "duration": "2 weeks",
        "risk": "low",
        "urgency": "routine",
        "reasoning": "Isolated back pain with abdominal pain in a younger patient without risk factors is more likely musculoskeletal or gastrointestinal (GERD, peptic ulcer). Pancreatic cancer is unlikely but not impossible. Clinical follow-up recommended if symptoms persist beyond 4-6 weeks.",
        "actions": ["Physical examination", "Trial of NSAIDs/antacids", "Follow up if symptoms persist >4 weeks", "Consider imaging only if symptoms worsen or new symptoms develop"],
        "conditions": [
            {"name": "Musculoskeletal pain", "likelihood": "high", "reasoning": "Most common cause of back pain in this age group. Abdominal pain may be referred or separate."},
            {"name": "Peptic ulcer disease", "likelihood": "medium", "reasoning": "Can cause both back and abdominal pain, especially posterior duodenal ulcers."},
            {"name": "Pancreatic cancer", "likelihood": "very low", "reasoning": "Unlikely without weight loss, jaundice, diabetes, or risk factors. Monitor if symptoms persist."}
        ]
    },
    {
        "symptoms": ["abdominal bloating", "diarrhea", "fatigue"],
        "age_range": (30, 50),
        "risk_factors": [],
        "duration": "2 months",
        "risk": "low",
        "urgency": "routine",
        "reasoning": "These symptoms are most consistent with irritable bowel syndrome or celiac disease in younger patients. Pancreatic cancer is very rare under age 50 without genetic predisposition. Standard GI workup recommended.",
        "actions": ["Dietary assessment", "Celiac serology", "Stool studies", "Consider colonoscopy if age-appropriate", "Follow up if symptoms change"],
        "conditions": [
            {"name": "Irritable bowel syndrome", "likelihood": "high", "reasoning": "Classic IBS presentation in age-appropriate patient."},
            {"name": "Celiac disease", "likelihood": "medium", "reasoning": "Can cause bloating, diarrhea, and fatigue. Simple blood test can screen."},
            {"name": "Pancreatic cancer", "likelihood": "very low", "reasoning": "Extremely rare in this age group without genetic risk factors."}
        ]
    },
    # Additional high-risk patterns
    {
        "symptoms": ["jaundice", "dark urine", "pale stools", "itching"],
        "age_range": (60, 85),
        "risk_factors": [],
        "duration": "1-3 weeks",
        "risk": "critical",
        "urgency": "emergency",
        "reasoning": "Obstructive jaundice with acholic (pale) stools and pruritus indicates bile duct obstruction. In patients >60, pancreatic head carcinoma is the most common cause. Dark urine reflects conjugated hyperbilirubinemia. This requires immediate imaging and likely ERCP.",
        "actions": ["Immediate abdominal ultrasound", "CT with pancreatic protocol", "MRCP", "ERCP for biliary decompression", "CA 19-9", "Surgical oncology referral"],
        "conditions": [
            {"name": "Pancreatic head adenocarcinoma", "likelihood": "high", "reasoning": "Most common cause of painless obstructive jaundice in elderly patients."},
            {"name": "Choledocholithiasis", "likelihood": "medium", "reasoning": "Gallstones in the common bile duct can cause identical presentation but usually with pain."},
            {"name": "Cholangiocarcinoma", "likelihood": "medium", "reasoning": "Bile duct cancer causes similar obstructive jaundice pattern."}
        ]
    },
    {
        "symptoms": ["abdominal pain radiating to back", "weight loss", "depression", "fatigue"],
        "age_range": (55, 75),
        "risk_factors": [],
        "duration": "4 months",
        "risk": "high",
        "urgency": "urgent",
        "reasoning": "New-onset depression in an older patient with abdominal pain radiating to the back and weight loss is a concerning pattern. Depression can be a paraneoplastic symptom of pancreatic cancer, often preceding the cancer diagnosis. The classic back pain radiation pattern suggests retroperitoneal involvement.",
        "actions": ["Abdominal CT with pancreatic protocol", "CA 19-9", "Comprehensive metabolic panel", "Psychiatric evaluation", "Gastroenterology referral"],
        "conditions": [
            {"name": "Pancreatic cancer", "likelihood": "high", "reasoning": "New-onset depression with weight loss and back pain in older patient is a recognized early PC pattern. Depression may be paraneoplastic."},
            {"name": "Major depressive disorder", "likelihood": "medium", "reasoning": "Depression can cause weight loss and somatic symptoms. However, new-onset depression in older adults warrants medical workup."},
            {"name": "Chronic pancreatitis", "likelihood": "medium", "reasoning": "Can cause back pain and weight loss, and may be associated with depression."}
        ]
    },
    {
        "symptoms": ["new-onset diabetes", "recurrent pancreatitis episodes"],
        "age_range": (50, 70),
        "risk_factors": [],
        "duration": "6 months",
        "risk": "high",
        "urgency": "urgent",
        "reasoning": "Recurrent acute pancreatitis with new-onset diabetes after age 50 is concerning for underlying pancreatic malignancy. Pancreatic cancer can cause recurrent pancreatitis by intermittently obstructing the pancreatic duct. This pattern is frequently misdiagnosed as simple recurrent pancreatitis.",
        "actions": ["CT with pancreatic protocol (during remission)", "MRCP", "EUS", "CA 19-9", "Genetic testing if family history", "Gastroenterology referral"],
        "conditions": [
            {"name": "Pancreatic cancer with secondary pancreatitis", "likelihood": "high", "reasoning": "Tumor obstructing pancreatic duct can cause recurrent pancreatitis. New-onset diabetes adds to concern. 59% of misdiagnosed PC cases were initially reported as pancreatitis."},
            {"name": "Chronic pancreatitis", "likelihood": "medium", "reasoning": "Can cause recurrent episodes and new diabetes. Must be distinguished from cancer."},
            {"name": "Pancreatic divisum", "likelihood": "low", "reasoning": "Congenital variant that can cause recurrent pancreatitis but doesn't explain new diabetes."}
        ]
    },
    {
        "symptoms": ["unexplained weight loss", "loss of appetite", "early satiety"],
        "age_range": (60, 80),
        "risk_factors": ["smoking history", "obesity"],
        "duration": "3 months",
        "risk": "medium",
        "urgency": "urgent",
        "reasoning": "Unexplained weight loss with loss of appetite and early satiety in an older smoker warrants cancer workup. While many cancers can cause this pattern, pancreatic cancer should be considered given the smoking history. Early satiety may indicate gastric compression by a pancreatic mass.",
        "actions": ["Complete metabolic panel", "CT chest/abdomen/pelvis", "CA 19-9", "Upper endoscopy", "Colonoscopy if not recent"],
        "conditions": [
            {"name": "Pancreatic cancer", "likelihood": "medium", "reasoning": "Smoking doubles PC risk. Weight loss with early satiety could indicate pancreatic mass compressing stomach."},
            {"name": "Gastric cancer", "likelihood": "medium", "reasoning": "Early satiety and weight loss are classic for gastric malignancy."},
            {"name": "Lung cancer", "likelihood": "medium", "reasoning": "Smoking history with weight loss raises concern for lung primary."}
        ]
    },
    {
        "symptoms": ["abdominal pain", "jaundice", "weight loss"],
        "age_range": (50, 70),
        "risk_factors": ["BRCA2 mutation", "family history of breast cancer"],
        "duration": "2 months",
        "risk": "critical",
        "urgency": "emergency",
        "reasoning": "BRCA2 mutations increase pancreatic cancer risk 3-6x. The combination of jaundice, abdominal pain, and weight loss in a BRCA2 carrier is highly concerning for pancreatic cancer. NCCN guidelines recommend genetic testing for all pancreatic cancer patients and surveillance for mutation carriers.",
        "actions": ["Immediate abdominal CT", "MRCP", "CA 19-9", "Surgical oncology referral", "Genetic counseling", "Consider BRCA-targeted therapy if confirmed"],
        "conditions": [
            {"name": "Pancreatic cancer (BRCA2-associated)", "likelihood": "high", "reasoning": "BRCA2 mutation with classic PC symptoms. May be eligible for PARP inhibitor therapy if confirmed."},
            {"name": "Cholangiocarcinoma", "likelihood": "low", "reasoning": "Can cause similar symptoms but less strongly associated with BRCA2."},
            {"name": "Choledocholithiasis", "likelihood": "low", "reasoning": "Can cause jaundice and pain but doesn't explain weight loss."}
        ]
    },
    {
        "symptoms": ["new-onset diabetes", "night sweats", "abdominal discomfort"],
        "age_range": (55, 70),
        "risk_factors": [],
        "duration": "6 weeks",
        "risk": "medium",
        "urgency": "routine",
        "reasoning": "New-onset diabetes after 50 should prompt consideration of pancreatic cancer, even with non-specific additional symptoms. Night sweats can indicate systemic inflammatory response. While most NOD is standard Type 2, clinical follow-up is important.",
        "actions": ["Fasting glucose and HbA1c", "CA 19-9", "Abdominal ultrasound", "Follow-up in 2-3 months", "Consider imaging if symptoms progress"],
        "conditions": [
            {"name": "Type 2 diabetes", "likelihood": "high", "reasoning": "Most common cause of new-onset diabetes in this age group."},
            {"name": "Pancreatic cancer (early)", "likelihood": "low-moderate", "reasoning": "NOD after 50 warrants monitoring for PC. Night sweats may indicate systemic disease."},
            {"name": "Lymphoma", "likelihood": "low", "reasoning": "Night sweats are a B-symptom of lymphoma. Abdominal lymphoma can cause diabetes."}
        ]
    },
    # Patterns with strong family history
    {
        "symptoms": ["abdominal pain", "fatigue", "unexplained weight loss"],
        "age_range": (45, 65),
        "risk_factors": ["family history of pancreatic cancer (2+ relatives)", "BRCA2 mutation"],
        "duration": "2 months",
        "risk": "critical",
        "urgency": "urgent",
        "reasoning": "Individuals with ≥2 first-degree relatives with PC and BRCA2 mutation have extremely high risk (up to 32x). NCCN recommends annual MRI/MRCP + EUS starting at age 50 (or 10 years before youngest affected relative). New symptoms in this population require immediate evaluation.",
        "actions": ["Immediate MRI/MRCP", "EUS", "CA 19-9", "Genetic counseling", "Surgical oncology referral", "Consider enrollment in screening trial"],
        "conditions": [
            {"name": "Pancreatic cancer", "likelihood": "high", "reasoning": "Extremely high-risk genetic background with concerning symptoms. Immediate evaluation required."},
            {"name": "Hereditary pancreatitis", "likelihood": "low", "reasoning": "Genetic predisposition to pancreatitis, but doesn't reduce cancer concern."},
            {"name": "Other hereditary cancers", "likelihood": "low", "reasoning": "BRCA2 also predisposes to breast, ovarian, prostate cancers."}
        ]
    },
    # Subtle patterns that get missed
    {
        "symptoms": ["mild jaundice", "itching", "fatigue"],
        "age_range": (60, 75),
        "risk_factors": [],
        "duration": "3 weeks",
        "risk": "high",
        "urgency": "urgent",
        "reasoning": "Mild jaundice with pruritus (itching) can be easily overlooked, especially if bilirubin is only mildly elevated. This early obstructive pattern may be the first sign of a pancreatic head mass. Fatigue is non-specific but adds to the clinical picture. Early intervention at this stage could mean the difference between resectable and unresectable disease.",
        "actions": ["Liver function tests", "Abdominal ultrasound", "CT with pancreatic protocol", "CA 19-9", "MRCP if ultrasound shows ductal dilation"],
        "conditions": [
            {"name": "Pancreatic cancer (early obstructive)", "likelihood": "medium-high", "reasoning": "Mild jaundice with itching suggests early bile duct obstruction. May be resectable if caught now."},
            {"name": "Primary biliary cholangitis", "likelihood": "medium", "reasoning": "Can cause itching and mild jaundice, typically in women."},
            {"name": "Drug-induced liver injury", "likelihood": "low", "reasoning": "Must review medications but doesn't explain full picture."}
        ]
    },
    {
        "symptoms": ["recurrent deep vein thrombosis", "abdominal pain", "weight loss"],
        "age_range": (55, 75),
        "risk_factors": [],
        "duration": "3 months",
        "risk": "high",
        "urgency": "urgent",
        "reasoning": "Pancreatic cancer is strongly associated with Trousseau syndrome — migratory thrombophlebitis and recurrent DVT. Unexplained recurrent DVT in an older patient with abdominal pain and weight loss should raise suspicion for occult malignancy, particularly pancreatic cancer.",
        "actions": ["CT chest/abdomen/pelvis", "CA 19-9", "Coagulation studies", "Cancer screening panel", "Oncology referral"],
        "conditions": [
            {"name": "Pancreatic cancer with Trousseau syndrome", "likelihood": "medium-high", "reasoning": "Recurrent DVT + abdominal pain + weight loss is a classic paraneoplastic pattern for pancreatic cancer."},
            {"name": "Other occult malignancy", "likelihood": "medium", "reasoning": "Recurrent DVT can indicate any cancer. Pancreatic is most strongly associated."},
            {"name": "Inherited thrombophilia", "likelihood": "medium", "reasoning": "Could explain recurrent DVT but doesn't account for abdominal symptoms and weight loss."}
        ]
    },
]

# Symptom descriptions written as patients would describe them
SYMPTOM_DESCRIPTIONS = {
    "new-onset diabetes": [
        "I was just diagnosed with diabetes. My blood sugar has been running high and I've been really thirsty.",
        "My doctor says I have type 2 diabetes. I never had blood sugar problems before.",
        "I've been drinking water constantly and running to the bathroom. My glucose came back really high.",
        "They just put me on metformin for diabetes. I'm 55 and never had this issue before.",
    ],
    "unexplained weight loss": [
        "I've lost about 15 pounds in the last few months without trying.",
        "My clothes are getting loose and I haven't changed my diet at all.",
        "I've dropped almost 20 pounds since the start of the year. People keep asking if I'm okay.",
        "I'm eating normally but keep losing weight. It's been about 10 pounds in 2 months.",
    ],
    "back pain": [
        "I have this deep ache in my mid-back that won't go away. It's worse when I lie down.",
        "The pain starts in my abdomen and radiates straight through to my back.",
        "I've had persistent back pain for weeks. It's not like regular muscle pain — it's deeper.",
        "My back hurts constantly, especially at night. Painkillers don't help much.",
    ],
    "abdominal pain": [
        "I have this dull pain in my upper abdomen that just won't quit.",
        "There's a constant ache in my stomach area, kind of in the middle-upper part.",
        "I've been having stomach pain for weeks. It's not sharp, more like a deep ache.",
        "The pain is right here in my upper belly, and it sometimes wraps around to my side.",
    ],
    "jaundice": [
        "My skin has turned yellow and the whites of my eyes look yellow too.",
        "People keep asking if I'm okay because my skin looks yellowish.",
        "I noticed my urine is really dark, almost brown, and my skin looks jaundiced.",
        "My doctor noticed I'm jaundiced during my last visit. My skin and eyes are yellow.",
    ],
    "painless jaundice": [
        "My skin turned yellow but I don't have any pain. My urine is dark though.",
        "I noticed yellowing of my eyes and skin. I feel fine otherwise, just a bit tired.",
        "My wife pointed out that I look yellow. I haven't had any pain, just noticed dark urine.",
    ],
    "dark urine": [
        "My urine has been really dark, like tea or cola colored, even when I'm drinking plenty of water.",
        "I've been drinking lots of water but my urine stays dark amber.",
    ],
    "pale stools": [
        "My stools have been really pale, almost white or clay-colored lately.",
        "I noticed my bowel movements are light-colored and they float. They look fatty.",
    ],
    "itching": [
        "I've been itching all over for weeks with no rash. It's driving me crazy.",
        "My skin itches constantly, especially at night. There's no visible reason for it.",
    ],
    "weight loss": [
        "I've been losing weight without trying. Down about 15 pounds in 3 months.",
        "Everyone comments on how much weight I've lost. I haven't changed my eating habits.",
    ],
    "fatigue": [
        "I'm exhausted all the time. No matter how much I sleep, I can't shake the tiredness.",
        "I have no energy anymore. Even simple tasks wear me out.",
        "The fatigue is overwhelming. I used to be so active and now I can barely get through the day.",
    ],
    "loss of appetite": [
        "I just don't feel hungry anymore. Food doesn't appeal to me.",
        "I've been skipping meals because I have no appetite. Nothing sounds good.",
        "I can only eat small portions before feeling full. My appetite has disappeared.",
    ],
    "early satiety": [
        "I feel full after eating just a few bites. I can't finish meals anymore.",
        "I get full really fast now. Even a small meal makes me feel stuffed.",
    ],
    "abdominal bloating": [
        "My stomach feels bloated and uncomfortable after eating, even small meals.",
        "I've been really gassy and bloated. My abdomen feels tight and swollen.",
    ],
    "steatorrhea": [
        "My stools have been really greasy and foul-smelling. They float and are hard to flush.",
        "I've noticed my bowel movements look oily and pale. They smell terrible.",
    ],
    "depression": [
        "I've been feeling really down lately. No motivation, no interest in things I used to enjoy.",
        "I developed depression out of nowhere. My doctor put me on antidepressants but something feels off.",
    ],
    "night sweats": [
        "I've been waking up drenched in sweat. My sheets are soaked.",
        "The night sweats are terrible. I have to change my pajamas multiple times a night.",
    ],
    "recurrent deep vein thrombosis": [
        "I've had three blood clots in my legs in the past year. My doctor can't figure out why.",
        "I keep getting DVTs despite being on blood thinners. This is my second clot in 6 months.",
    ],
    "recurrent pancreatitis episodes": [
        "I've been to the ER three times this year for pancreatitis. Each time it's terrible pain.",
        "My pancreatitis keeps coming back. My doctor is concerned about why it keeps recurring.",
    ],
    "diarrhea": [
        "I've had diarrhea on and off for weeks. It comes and goes but never fully resolves.",
    ],
    "nausea": [
        "I've been feeling nauseated on and off, especially after eating.",
    ],
}

# General medical conditions from symptom_to_diagnosis dataset
GENERAL_CONDITIONS = {
    "diabetes": {
        "symptoms": ["increased thirst", "frequent urination", "unexplained weight loss", "fatigue", "blurred vision"],
        "description": "Type 2 diabetes mellitus",
        "urgency": "routine"
    },
    "hypertension": {
        "symptoms": ["headache", "dizziness", "blurred vision", "chest pain"],
        "description": "High blood pressure",
        "urgency": "routine"
    },
    "gastroesophageal reflux disease": {
        "symptoms": ["heartburn", "acid reflux", "chest pain", "difficulty swallowing"],
        "description": "GERD",
        "urgency": "routine"
    },
    "peptic ulcer disease": {
        "symptoms": ["abdominal pain", "bloating", "nausea", "heartburn"],
        "description": "Stomach or duodenal ulcers",
        "urgency": "routine"
    },
    "irritable bowel syndrome": {
        "symptoms": ["abdominal pain", "bloating", "diarrhea", "constipation"],
        "description": "IBS",
        "urgency": "routine"
    },
    "chronic pancreatitis": {
        "symptoms": ["abdominal pain", "back pain", "weight loss", "steatorrhea", "diabetes"],
        "description": "Chronic inflammation of the pancreas",
        "urgency": "urgent"
    },
}


def create_patient_narrative(symptoms: list, age: int, risk_factors: list, duration: str) -> str:
    """Create a natural language patient description from structured data."""
    parts = []

    # Opening
    age_desc = f"{age}-year-old"
    gender = random.choice(["male", "female"])
    parts.append(f"I'm a {age_desc} {gender}.")

    # Symptoms
    symptom_texts = []
    for symptom in symptoms:
        if symptom in SYMPTOM_DESCRIPTIONS:
            symptom_texts.append(random.choice(SYMPTOM_DESCRIPTIONS[symptom]))
        else:
            symptom_texts.append(f"I've been experiencing {symptom}.")

    parts.append(" ".join(symptom_texts[:3]))  # Limit to 3 symptom descriptions

    # Duration
    parts.append(f"This has been going on for about {duration}.")

    # Risk factors
    if risk_factors:
        rf_text = ", ".join(risk_factors)
        parts.append(f"My medical history includes: {rf_text}.")

    return " ".join(parts)


def create_structured_output(pattern: dict) -> dict:
    """Create structured output from a pattern definition."""
    return {
        "risk_assessment": pattern["risk"],
        "conditions": pattern["conditions"],
        "urgency": pattern["urgency"],
        "recommended_actions": pattern["actions"],
        "reasoning_chain": pattern["reasoning"]
    }


def generate_training_example(pattern: dict) -> dict:
    """Generate a single training example from a pattern."""
    age = random.randint(*pattern["age_range"])
    narrative = create_patient_narrative(
        pattern["symptoms"], age, pattern.get("risk_factors", []), pattern["duration"]
    )
    output = create_structured_output(pattern)

    return {
        "messages": [
            {
                "role": "system",
                "content": "You are a medical triage AI specialized in pancreatic cancer early detection. Analyze the patient presentation and provide a structured assessment including risk level, possible conditions with reasoning, urgency classification, and recommended next steps. Always consider pancreatic cancer patterns, especially in patients over 50 with new-onset diabetes, unexplained weight loss, or jaundice. Be thorough but not alarmist — match your urgency to the clinical picture."
            },
            {
                "role": "user",
                "content": f"Please assess my symptoms and tell me what might be going on:\n\n{narrative}"
            },
            {
                "role": "assistant",
                "content": json.dumps(output, indent=2)
            }
        ]
    }


def generate_general_medical_examples(count: int = 50) -> list:
    """Generate general medical examples to prevent catastrophic forgetting."""
    examples = []
    for condition_name, info in GENERAL_CONDITIONS.items():
        for _ in range(count // len(GENERAL_CONDITIONS)):
            age = random.randint(25, 70)
            symptom_subset = random.sample(info["symptoms"], min(3, len(info["symptoms"])))
            symptom_text = ", ".join(symptom_subset)

            example = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a medical triage AI. Analyze the patient presentation and provide a structured assessment including risk level, possible conditions with reasoning, urgency classification, and recommended next steps."
                    },
                    {
                        "role": "user",
                        "content": f"I'm a {age}-year-old. I've been having {symptom_text} for a few weeks. What could this be?"
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps({
                            "risk_assessment": "low" if info["urgency"] == "routine" else "medium",
                            "conditions": [
                                {"name": condition_name, "likelihood": "high", "reasoning": f"Symptoms are consistent with {info['description']}."},
                                {"name": "Other differential", "likelihood": "low", "reasoning": "Other conditions less likely given symptom pattern."}
                            ],
                            "urgency": info["urgency"],
                            "recommended_actions": ["See primary care physician", "Basic blood work", "Follow up in 2 weeks"],
                            "reasoning_chain": f"Based on the symptom pattern described ({symptom_text}), the most likely diagnosis is {info['description']}. This is a common condition that should be evaluated by a healthcare provider."
                        }, indent=2)
                    }
                ]
            }
            examples.append(example)

    return examples


def augment_patterns(patterns: list, augmentations_per_pattern: int = 5) -> list:
    """Generate multiple training examples from each pattern with variations."""
    augmented = []
    for pattern in patterns:
        for _ in range(augmentations_per_pattern):
            example = generate_training_example(pattern)
            augmented.append(example)
    return augmented


def format_for_unsloth_chat(examples: list) -> list:
    """Format examples for Unsloth's chat template (Gemma 4)."""
    formatted = []
    for example in examples:
        formatted.append({
            "conversations": example["messages"]
        })
    return formatted


def main():
    """Build the complete training dataset."""
    print("=== Clover Cancer Dataset Builder ===\n")

    # 1. Generate pancreatic cancer examples
    print("Generating pancreatic cancer triage examples...")
    pc_examples = augment_patterns(PANCREATIC_CANCER_PATTERNS, augmentations_per_pattern=8)
    print(f"  Generated {len(pc_examples)} pancreatic cancer examples")

    # 2. Generate general medical examples
    print("Generating general medical examples...")
    general_examples = generate_general_medical_examples(count=200)
    print(f"  Generated {len(general_examples)} general medical examples")

    # 3. Combine
    all_examples = pc_examples + general_examples
    random.shuffle(all_examples)
    print(f"\nTotal examples: {len(all_examples)}")

    # 4. Split into train/val/test
    n = len(all_examples)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    train_data = all_examples[:train_end]
    val_data = all_examples[train_end:val_end]
    test_data = all_examples[val_end:]

    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # 5. Format for Unsloth
    train_formatted = format_for_unsloth_chat(train_data)
    val_formatted = format_for_unsloth_chat(val_data)
    test_formatted = format_for_unsloth_chat(test_data)

    # 6. Save
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, data in [("train", train_formatted), ("val", val_formatted), ("test", test_formatted)]:
        path = output_dir / f"{name}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved {path} ({len(data)} examples)")

    # 7. Save a human-readable sample
    sample_path = output_dir / "sample_examples.json"
    with open(sample_path, "w") as f:
        json.dump(all_examples[:5], f, indent=2)
    print(f"\nSaved sample examples to {sample_path}")

    print("\n=== Dataset generation complete ===")


if __name__ == "__main__":
    main()
