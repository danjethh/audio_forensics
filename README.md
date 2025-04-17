##  Audio Tampering Detection Using Unsupervised Machine Learning

##  Problem Statement

In the field of digital forensics, **audio evidence** plays a crucial role in criminal investigations, legal proceedings, intelligence gathering, and whistleblower protection. However, the integrity of audio recordings is increasingly under threat due to the rise of sophisticated editing tools and artificial voice synthesis technologies.

**Audio tampering** refers to the deliberate manipulation of a recording's content—ranging from the removal or insertion of words, to seamless splicing of segments or injection of misleading background sounds. These modifications are often so subtle that they evade detection by the naked ear or basic waveform inspection.

###  Challenges Faced by Forensic Investigators

- **Lack of Ground Truth**: In many cases, investigators lack access to the original unaltered recording, making it difficult to verify authenticity.
- **Sophistication of Editing Tools**: Modern audio editors can manipulate recordings without leaving clear spectral or temporal artifacts.
- **Time-Intensive Manual Review**: Traditional methods like spectrogram analysis, noise floor monitoring, and Electric Network Frequency (ENF) analysis are labor-intensive and require expert interpretation.
- **Device and Environment Variability**: Variations in microphones, environments, and compression settings can introduce false positives or mask genuine tampering.
- **Inconsistent Metadata**: Relying on file properties (e.g., encoding timestamps, bitrate) can be unreliable, as metadata itself can be forged or stripped.

###  Existing and Traditional Detection Methods

Forensic analysts have historically employed methods such as:

- **Spectral analysis** to identify unnatural breaks or anomalies in frequency content
- **ENF analysis** to detect inconsistencies in the electrical hum embedded in the recording
- **Waveform continuity checks** to catch abrupt transitions or duplicated segments
- **Acoustic environment comparison** to detect changes in background reverberation or noise signatures
- **Manual auditory inspection**, which is subjective, time-consuming, and error-prone

While effective in isolated cases, these approaches do not scale well when triaging large volumes of evidence or detecting subtle forms of tampering.

> This project addresses the need for a scalable, automated solution that can act as a first line of defense in flagging potentially altered audio—using only basic, machine-extractable features.


##  Why Use AI for Audio Tampering Detection?

Traditional signal processing techniques have long supported audio forensics, but they often fall short when dealing with subtle, complex, or deliberately concealed tampering. Artificial Intelligence (AI), particularly unsupervised machine learning, offers a scalable and intelligent alternative:

1. **Beyond Human Perception**  
   AI models detect patterns and inconsistencies in waveform data that are too subtle for human ears or rule-based systems to reliably identify.

2. **No Labels Required**  
   In real-world investigations, labeled tampered audio is rare. Unsupervised models like Isolation Forest learn from normal data alone, flagging deviations without prior examples of every attack.

3. **Robust to Variability**  
   AI generalizes across different speakers, environments, microphones, and tampering types—making it suitable for diverse forensic scenarios.

4. **Fast and Scalable**  
   AI can process thousands of files in seconds, far outperforming manual analysis and traditional tools in both speed and volume handling.

> In short: AI helps investigators detect what’s invisible, automate what’s tedious, and scale what’s otherwise impractical.

---

##  Lab Outcomes

By the end of this lab-based implementation, the following milestones were achieved:

1. **Unsupervised Anomaly Detection**  
   Successfully applied the Isolation Forest algorithm to identify abnormal patterns in unlabeled audio data.

2. **Lightweight Feature Representation**  
   Used minimal yet meaningful features—\texttt{Duration} and \texttt{Sample Rate}—ensuring low computational cost.

3. **Tampering Identification**  
   Accurately flagged structurally anomalous audio (e.g., unusual durations) as potentially tampered.

4. **Visual Interpretation**  
   Displayed anomaly prediction distributions to make the model’s decisions transparent and intuitive.

5. **Forensic Applicability**  
   Demonstrated how AI can be used in real forensic workflows, especially in the absence of labeled datasets or overt manipulation clues.


##  Goal of the Lab

The goal of this lab is to develop an AI-powered system that can detect potentially tampered audio recordings using a lightweight, unsupervised machine learning approach. Specifically, it leverages the **Isolation Forest** algorithm trained on a minimal set of acoustic features.

This system is designed to:

-  Automatically identify audio recordings that may have been manipulated or structurally altered.
-  Operate without the need for labeled training data (unsupervised learning).
-  Use computationally efficient features such as **Duration** and **Sample Rate** for fast and scalable performance.
-  Lay the groundwork for real-world applications in forensic investigations, evidence verification, and digital security.

---

##  Dataset Source

The dataset used in this project is sourced from a publicly available audio forensic dataset hosted on GitHub. It contains a collection of untampered (genuine) audio recordings, each represented by several derived acoustic features.

- **GitHub Dataset**: [Audio Forensics Dataset](https://raw.githubusercontent.com/danjethh/audio_forensics/main/Values.csv.xls)

You can load this dataset using Python’s `pandas` library:
```python
import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/danjethh/audio_forensics/main/Values.csv.xls")
```

### Step 2: Clean Dataset and Select Relevant Features

 **What happened?**  
From the 27 available features, only two were selected for model training:

- `Duration (ss)`
- `Sample Rate (Hz)`

These features were chosen because they are computationally simple and can be directly extracted from any `.wav` file using tools like [`librosa`](https://librosa.org/). To ensure data integrity:

- All non-relevant columns were dropped  
- Rows with missing values were excluded  
- The cleaned dataset was saved as `cleaned_audio_dataset.csv`

 **Why these features?**

- **Simplicity** – Reduces computational overhead and risk of overfitting  
- **Universality** – Easily extractable across diverse file types and devices  
- **Forensic Relevance** – Even subtle tampering can unintentionally affect duration or sample rate

 **Data Cleaning Summary:**

-  614 total audio recordings retained  
-  No missing values in selected columns  
-  Cleaned dataset stored locally for training


### Step 3: Train Isolation Forest Model

 **What happened?**  
An Isolation Forest model was trained on the cleaned feature matrix consisting of `Duration` and `Sample Rate`. This unsupervised learning algorithm builds multiple random trees and isolates data points that differ significantly from the rest.

- Trained on 611 audio samples
- Automatically learned the “normal” distribution of untampered audio
- Labeled statistically unusual samples as anomalies

 **Why use Isolation Forest?**

- Ideal when we have many examples of normal data, but no labeled anomalies  
- Makes no assumption about the underlying data distribution  
- Efficient for high-dimensional, noisy, or unstructured input  
- Scales well to large datasets

 **Output Summary:**

- Model trained using 2 core features  
- 183 out of 614 recordings flagged as anomalies (likely due to duration or sampling irregularities)


### Step 4: Save Model and Visualize Prediction Results

 **What happened?**  
- The trained model was saved as `audio_isolation_forest_model.pkl` for future reuse.
- A bar chart was generated to visualize the model’s output:
  - `1` → Normal audio  
  - `-1` → Anomalous (outlier) audio

 **Why this matters:**  
- Saving the model avoids retraining each time, enabling efficient deployment.
- Visualization helps assess the model's behavior:
  - Too many anomalies → overly sensitive  
  - Too few → possibly too lenient

This diagnostic step ensures that the anomaly detector is balanced and reliable for forensic use.


### Step 5: Extract Features from Suspicious Audio File

 **What happened?**  
- A test audio file (hosted on GitHub) was downloaded for evaluation.
- Its `Duration` and `Sample Rate` were extracted using the `librosa` audio analysis library.

 **Why this matters:**  
- To make a valid prediction, we must extract the same features used during training.
- This maintains consistency and ensures the model can interpret the new audio input correctly.

 **Output Example:**
| Duration (ss) | Sample Rate (Hz) |
|---------------|------------------|
| 3.24          | 44100            |

This new feature vector is then passed to the trained model to determine if the file is anomalous.


### Step 6: Predict If the Audio is Anomalous

 **What happened?**  
- The extracted features from the suspicious audio file were passed into the trained Isolation Forest model.
- The model returned a prediction:
  - `1` → Normal (conforms to training distribution)
  - `-1` → Anomalous (statistically different from known clean data)
- The result was presented in both numeric and plain English formats.

 **Why this matters:**  
This step is the core purpose of the system: to determine whether a new, unknown audio file shows signs of tampering or manipulation based on learned statistical patterns from clean audio.

 **Output:**

If the model predicts `-1`:
> “The model flagged this audio as **ANOMALOUS** (possibly tampered).”

If the model predicts `1`:
> “The model flagged this audio as **NORMAL** (no anomalies detected).”

In this case study, the model flagged the suspicious audio file as **ANOMALOUS** (`-1`), indicating that its characteristics deviated significantly from the baseline established during training.
