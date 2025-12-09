import streamlit as st
import onnxruntime
import numpy as np
import os

# --- Configuration ---
# Get the directory of the current script to build a relative path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# The model is located one directory level up from the app's directory
MODEL_PATH = os.path.join(_SCRIPT_DIR, "utils", "lasso_predictor.onnx")


# --- Helper Functions ---

@st.cache_resource
def load_model(model_path):
    """
    Loads the ONNX model and creates an inference session.
    Using @st.cache_resource to load the model only once.
    """
    try:
        session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        return session
    except FileNotFoundError:
        st.error(f"🔴 Model file not found at '{model_path}'. Please make sure the model is in the same directory as the app.")
        return None
    except Exception as e:
        st.error(f"🔴 An error occurred while loading the model: {e}")
        return None

def get_model_details(_session):
    """
    Retrieves and displays the input/output details of the loaded model.
    """
    if _session:
        inputs = _session.get_inputs()
        outputs = _session.get_outputs()
        with st.expander("🔬 Model Details"):
            st.write("**Inputs:**")
            for inp in inputs:
                st.write(f"- Name: `{inp.name}`, Shape: `{inp.shape}`, Type: `{inp.type}`")
            st.write("**Outputs:**")
            for out in outputs:
                st.write(f"- Name: `{out.name}`, Shape: `{out.shape}`, Type: `{out.type}`")

# --- Streamlit App UI ---

st.set_page_config(layout="wide", page_title="Penicillin Yield Predictor", page_icon="💊")

# --- Custom CSS for Presentation Mode ---
st.markdown("""
    <style>
    /* Global font size increase */
    html, body, [class*="css"]  {
        font-size: 22px;
    }
    
    /* Headers */
    h1 { font-size: 4rem !important; }
    h2 { font-size: 3rem !important; }
    h3 { font-size: 2.5rem !important; }
    
    /* Text elements */
    p { font-size: 1.4rem !important; }
    
    /* Widget labels */
    .stNumberInput label p, .stSelectbox label p {
        font-size: 1.5rem !important;
    }
    
    /* Input fields */
    .stNumberInput input {
        font-size: 1.5rem !important;
    }
    
    /* Buttons */
    .stButton button {
        font-size: 1.8rem !important;
        padding: 0.5rem 1rem !important;
    }
    
    /* Expander headers */
    .streamlit-expanderHeader p {
        font-size: 1.6rem !important;
    }
    
    /* Alert boxes (info, success, error) */
    .stAlert {
        font-size: 1.4rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Header ---
st.title("💊 Penicillin Yield Predictor")
st.markdown("An interactive demo to predict penicillin yield using a pre-trained ONNX model. Adjust the sliders and see the prediction change in real-time!")

# Add Lasso model explanation
with st.expander("🤖 About the Lasso Regression Model", expanded=False):
    st.markdown("""
    ### What is Lasso Regression? 🎯

    **LASSO** (Least Absolute Shrinkage and Selection Operator) is a linear regression technique that performs both **prediction** and **feature selection** simultaneously.

    **Key Characteristics:**
    - **Linear Model**: Predicts yield as a weighted sum of input features
    - **Feature Selection**: Automatically identifies the most important features
    - **Regularization**: Prevents overfitting by penalizing complex models
    - **Interpretable**: Easy to understand which features drive predictions

    ---

    ### How LASSO Works 🔧

    The model minimizes this equation:
    
    **Loss = Mean Squared Error + λ × |coefficients|**

    Where:
    - **Mean Squared Error**: Measures prediction accuracy
    - **λ (lambda)**: Controls regularization strength
    - **|coefficients|**: Sum of absolute values of feature weights

    **The Magic of L1 Regularization:**
    - Forces some feature coefficients to exactly **zero**
    - Automatically **removes irrelevant features** from the model
    - Creates **sparse models** with only important predictors
    - Balances **accuracy** vs **simplicity**

    ---

    ### Why LASSO for Penicillin Production? 🧬

    **✅ Advantages in Bioprocessing:**

    1. **Feature Selection**: Automatically identifies which process parameters truly matter
    2. **Interpretability**: Clear understanding of parameter impacts
    3. **Robustness**: Handles noisy sensor data well
    4. **Efficiency**: Fast predictions suitable for real-time monitoring
    5. **Prevents Overfitting**: Works well with limited training data

    **🎯 Model Performance:**
    - Trained on **1,439 fermentation samples**
    - Selected **14 key temporal features** from larger feature set
    - Optimized λ parameter through cross-validation
    - Balances prediction accuracy with model simplicity

    **📊 Feature Importance Discovery:**
    The LASSO automatically discovered that **Offline Biomass** and **Carbon Evolution Rate** 
    are the most predictive features, while substrate and chemical concentrations have minimal impact.
    This aligns with biological understanding of fermentation processes!
    """)

# Add explanatory info box
with st.expander("ℹ️ Understanding Temporal Features & Data Drift", expanded=False):
    st.markdown("""
    ### Why Two Values Per Parameter? ⏱️

    This model uses **temporal features** - it considers measurements from two consecutive time points `[0]` and `[1]`
    to capture the **rate of change** in the fermentation process. This temporal information helps the model understand:

    - **Trends**: Is biomass increasing or decreasing?
    - **Process dynamics**: How quickly are conditions changing?
    - **Production stage**: Different phases show different temporal patterns

    By comparing values at two time points, the model can better predict future yield than using a single snapshot.

    ---

    ### Feature Impact on Yield Prediction 🎯

    Based on correlation analysis of 1,439 samples:

    **🔴 HIGH IMPACT Features** (Strong predictors):
    - **Offline Biomass** (correlation: 0.79) - Most important feature
    - **Carbon Evolution Rate** (correlation: 0.72) - Second most important

    **🟡 MEDIUM IMPACT Features** (Moderate predictors):
    - **Oxygen Uptake Rate** (correlation: 0.21)

    **🟢 LOW IMPACT Features** (Weak predictors):
    - Substrate Concentration (correlation: 0.11)
    - NH3 Concentration (correlation: 0.06)
    - PAA Concentration (correlation: 0.04)

    ---

    ### Data Drift in Production 📊

    **Data drift** occurs when production data distribution shifts from training data, degrading model performance.

    **Features Most Susceptible to Drift** (high variability):
    - **Substrate Concentration** (CV: 347%) - Extremely variable
    - **PAA Concentration** (CV: 83%) - Highly variable

    **More Stable Features** (medium variability):
    - Carbon Evolution Rate (CV: 39%)
    - Offline Biomass (CV: 37%)
    - Oxygen Uptake Rate (CV: 37%)
    - NH3 Concentration (CV: 26%)

    **⚠️ Production Insight**: Monitor high-impact, high-drift features (Substrate, PAA) carefully,
    as they combine prediction importance with measurement instability.
    """)

st.markdown("---")

# --- Load Model and Display Details ---
session = load_model(MODEL_PATH)
if session:
    # Model details section removed for cleaner UI

    st.markdown("---")

    # --- Main Layout (Inputs and Outputs side-by-side) ---
    col1, col2 = st.columns([1, 1]) # Create two columns of equal width

    with col1:
        st.header("⚙️ Input Features")
        st.write("Use the controls below to set the penicillin yield features.")
        st.info("🔴 = High impact | 🟡 = Medium impact | 🟢 = Low impact")

        # Define default values
        defaults = {
            "cer_0": 1.4006, "nh3_0": 1786.3000, "biomass_0": 21.4870, 
            "our_0": 1.2645, "paa_0": 1201.3000, "substrate_0": 0.0016,
            "cer_1": 1.4006, "nh3_1": 1786.3000, "biomass_1": 21.4870, 
            "our_1": 1.2645, "paa_1": 1201.3000, "substrate_1": 0.0016
        }

        # Initialize session state
        for key, default_val in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_val

        # Reset button
        if st.button("🔄 Reset to Default Values", use_container_width=True, key="reset_button"):
            for key, value in defaults.items():
                st.session_state[key] = value
            st.rerun()

        # ==============================================================================
        # TODO: CUSTOMIZE YOUR INPUTS HERE
        # This section is hardcoded for a better user experience.
        # You must replace these with the inputs your specific model requires.
        # ==============================================================================

        # Create two columns for the inputs to make the layout cleaner
        input_col1, input_col2 = st.columns(2)

        with input_col1:
            cer_0 = st.number_input("🔴 Carbon Evolution Rate (g/h) [0]", format="%.4f", key="cer_0",
                                    help="High impact feature (corr: 0.72) - Monitors microbial metabolic activity")
            nh3_0 = st.number_input("🟢 NH3 Concentration (g/L) [0]", format="%.4f", key="nh3_0",
                                    help="Low impact feature (corr: 0.06) - Nitrogen source tracking")
            biomass_0 = st.number_input("🔴 Offline Biomass (g/L) [0]", format="%.4f", key="biomass_0",
                                        help="Highest impact feature (corr: 0.79) - Cell concentration measurement")
            # Hidden input - hardcoded value for Offline Penicillin [0]
            penicillin_0 = 14.5760
            our_0 = st.number_input("🟡 Oxygen Uptake Rate (g/min) [0]", format="%.4f", key="our_0",
                                    help="Medium impact feature (corr: 0.21) - Respiration rate")
            paa_0 = st.number_input("🟢 PAA Concentration (g/L) [0]", format="%.4f", key="paa_0",
                                    help="Low impact, high drift (CV: 83%) - Precursor compound")
            substrate_0 = st.number_input("🟢 Substrate Concentration (g/L) [0]", format="%.4f", key="substrate_0",
                                          help="Low impact, extremely high drift (CV: 347%) - Nutrient level")

        with input_col2:
            cer_1 = st.number_input("🔴 Carbon Evolution Rate (g/h) [1]", format="%.4f", key="cer_1",
                                    help="High impact feature (corr: 0.72) - Monitors microbial metabolic activity")
            nh3_1 = st.number_input("🟢 NH3 Concentration (g/L) [1]", format="%.4f", key="nh3_1",
                                    help="Low impact feature (corr: 0.06) - Nitrogen source tracking")
            biomass_1 = st.number_input("🔴 Offline Biomass (g/L) [1]", format="%.4f", key="biomass_1",
                                        help="Highest impact feature (corr: 0.79) - Cell concentration measurement")
            # Hidden input - hardcoded value for Offline Penicillin [1]
            penicillin_1 = 14.5760
            our_1 = st.number_input("🟡 Oxygen Uptake Rate (g/min) [1]", format="%.4f", key="our_1",
                                    help="Medium impact feature (corr: 0.21) - Respiration rate")
            paa_1 = st.number_input("🟢 PAA Concentration (g/L) [1]", format="%.4f", key="paa_1",
                                    help="Low impact, high drift (CV: 83%) - Precursor compound")
            substrate_1 = st.number_input("🟢 Substrate Concentration (g/L) [1]", format="%.4f", key="substrate_1",
                                          help="Low impact, extremely high drift (CV: 347%) - Nutrient level")

    with col2:
        st.header("📈 Prediction Output")
        st.write("The model's predicted yield will appear below.")
        
        # Predict button at the top of the output column
        predict_button = st.button("⚡ Predict Yield", use_container_width=True)

        if predict_button:
            try:
                # ==============================================================================
                # TODO: PREPARE YOUR INPUTS FOR THE MODEL
                # You must match the input names, data types, and shapes that your
                # model expects. The input names can be found in the "Model Details" expander.
                # ==============================================================================

                # This model expects multiple named inputs, not a single array.
                # We will create a dictionary mapping input names to their values.
                output_name = session.get_outputs()[0].name
                input_names = [inp.name for inp in session.get_inputs()]

                # 1. Collect all user inputs into a list in the correct order
                input_values = [
                    cer_0, cer_1,
                    nh3_0, nh3_1,
                    biomass_0, biomass_1,
                    penicillin_0, penicillin_1,
                    our_0, our_1,
                    paa_0, paa_1,
                    substrate_0, substrate_1
                ]

                # 2. Create the input dictionary for the model
                # Each input value needs to be a numpy array of shape [1, 1] and type float64 (double).
                input_feed = {
                    name: np.array([[value]], dtype=np.float64)
                    for name, value in zip(input_names, input_values)
                }

                # Run the prediction
                prediction = session.run([output_name], input_feed)[0]
                predicted_yield = prediction[0][0]

                # --- Display the result in a visually appealing way ---
                st.markdown(
                    f"""
                    <div style="
                        border: 2px solid #28a745;
                        border-radius: 10px;
                        padding: 20px;
                        text-align: center;
                        background-color: #f0fff0;">
                        <h3 style="color: #28a745; margin:0;">Predicted Yield</h3>
                        <p style="font-size: 2.5em; font-weight: bold; color: #2E8B57; margin:0;">
                            {predicted_yield:,.2f} g/L
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                # st.balloons()

            except Exception as e:
                st.error(f"🔴 An error occurred during prediction: {e}")

# --- Data Drift Demonstration Section ---
st.markdown("---")
st.header("🔬 Data Drift Demonstration")
st.markdown("""
Experiment with feature changes to understand their impact on predictions and observe how drift affects model behavior.
Try modifying the **high-impact features** (🔴) vs **low-impact features** (🟢) to see the difference!
""")

drift_col1, drift_col2 = st.columns(2)

with drift_col1:
    st.subheader("📊 Scenario Comparison")

    scenario = st.selectbox(
        "Select a drift scenario:",
        [
            "Baseline (Median values)",
            "High Biomass Growth (+50%)",
            "Low Biomass Growth (-50%)",
            "High Carbon Evolution (+50%)",
            "Substrate Depletion (-90%)",
            "Combined High-Impact Drift",
            "Metabolic Stress (High CER, Low Biomass)",
            "Metabolic Disconnect (High Biomass, Low CER)",
            "Process Failure (Low Activity)",
            "Sensor Failure (Substrate Spike)",
            "Late Stage Fermentation"
        ]
    )

    # Define scenarios
    scenarios = {
        "Baseline (Median values)": {
            "biomass_mult": 1.0, "cer_mult": 1.0, "substrate_mult": 1.0,
            "paa_mult": 1.0, "our_mult": 1.0, "nh3_mult": 1.0
        },
        "High Biomass Growth (+50%)": {
            "biomass_mult": 1.5, "cer_mult": 1.0, "substrate_mult": 1.0,
            "paa_mult": 1.0, "our_mult": 1.0, "nh3_mult": 1.0
        },
        "Low Biomass Growth (-50%)": {
            "biomass_mult": 0.5, "cer_mult": 1.0, "substrate_mult": 1.0,
            "paa_mult": 1.0, "our_mult": 1.0, "nh3_mult": 1.0
        },
        "High Carbon Evolution (+50%)": {
            "biomass_mult": 1.0, "cer_mult": 1.5, "substrate_mult": 1.0,
            "paa_mult": 1.0, "our_mult": 1.0, "nh3_mult": 1.0
        },
        "Substrate Depletion (-90%)": {
            "biomass_mult": 1.0, "cer_mult": 1.0, "substrate_mult": 0.1,
            "paa_mult": 1.0, "our_mult": 1.0, "nh3_mult": 1.0
        },
        "Combined High-Impact Drift": {
            "biomass_mult": 1.5, "cer_mult": 1.5, "substrate_mult": 1.0,
            "paa_mult": 1.0, "our_mult": 1.0, "nh3_mult": 1.0
        },
        "Metabolic Stress (High CER, Low Biomass)": {
            "biomass_mult": 0.8, "cer_mult": 1.5, "substrate_mult": 1.0,
            "paa_mult": 1.0, "our_mult": 1.0, "nh3_mult": 1.0
        },
        "Metabolic Disconnect (High Biomass, Low CER)": {
            "biomass_mult": 1.5, "cer_mult": 0.5, "substrate_mult": 1.0,
            "paa_mult": 1.0, "our_mult": 1.0, "nh3_mult": 1.0
        },
        "Process Failure (Low Activity)": {
            "biomass_mult": 0.5, "cer_mult": 0.5, "substrate_mult": 1.0,
            "paa_mult": 1.0, "our_mult": 0.5, "nh3_mult": 1.0
        },
        "Sensor Failure (Substrate Spike)": {
            "biomass_mult": 1.0, "cer_mult": 1.0, "substrate_mult": 10.0,
            "paa_mult": 1.0, "our_mult": 1.0, "nh3_mult": 1.0
        },
        "Late Stage Fermentation": {
            "biomass_mult": 1.0, "cer_mult": 1.0, "substrate_mult": 0.1,
            "paa_mult": 1.5, "our_mult": 0.8, "nh3_mult": 1.0
        }
    }

    st.markdown(f"**Selected scenario:** `{scenario}`")

    if scenario != "Baseline (Median values)":
        changes = []
        mult = scenarios[scenario]
        if mult["biomass_mult"] != 1.0:
            changes.append(f"- Biomass: {(mult['biomass_mult']-1)*100:+.0f}%")
        if mult["cer_mult"] != 1.0:
            changes.append(f"- Carbon Evolution Rate: {(mult['cer_mult']-1)*100:+.0f}%")
        if mult["substrate_mult"] != 1.0:
            changes.append(f"- Substrate: {(mult['substrate_mult']-1)*100:+.0f}%")
        if mult["paa_mult"] != 1.0:
            changes.append(f"- PAA: {(mult['paa_mult']-1)*100:+.0f}%")

        st.markdown("**Changes from baseline:**\n" + "\n".join(changes))

with drift_col2:
    st.subheader("🎯 Impact Prediction")

    if st.button("🔄 Run Drift Scenario", use_container_width=True):
        if session:
            try:
                mult = scenarios[scenario]

                # Apply multipliers to baseline values
                drift_cer_0 = 1.4006 * mult["cer_mult"]
                drift_cer_1 = 1.4006 * mult["cer_mult"]
                drift_nh3_0 = 1786.3000 * mult["nh3_mult"]
                drift_nh3_1 = 1786.3000 * mult["nh3_mult"]
                drift_biomass_0 = 21.4870 * mult["biomass_mult"]
                drift_biomass_1 = 21.4870 * mult["biomass_mult"]
                drift_penicillin_0 = 14.5760
                drift_penicillin_1 = 14.5760
                drift_our_0 = 1.2645 * mult["our_mult"]
                drift_our_1 = 1.2645 * mult["our_mult"]
                drift_paa_0 = 1201.3000 * mult["paa_mult"]
                drift_paa_1 = 1201.3000 * mult["paa_mult"]
                drift_substrate_0 = 0.0016 * mult["substrate_mult"]
                drift_substrate_1 = 0.0016 * mult["substrate_mult"]

                # Prepare model input
                output_name = session.get_outputs()[0].name
                input_names = [inp.name for inp in session.get_inputs()]

                input_values = [
                    drift_cer_0, drift_cer_1,
                    drift_nh3_0, drift_nh3_1,
                    drift_biomass_0, drift_biomass_1,
                    drift_penicillin_0, drift_penicillin_1,
                    drift_our_0, drift_our_1,
                    drift_paa_0, drift_paa_1,
                    drift_substrate_0, drift_substrate_1
                ]

                input_feed = {
                    name: np.array([[value]], dtype=np.float64)
                    for name, value in zip(input_names, input_values)
                }

                # Run prediction
                prediction = session.run([output_name], input_feed)[0]
                predicted_yield = prediction[0][0]

                # Calculate baseline for comparison
                baseline_input_values = [
                    1.4006, 1.4006,
                    1786.3000, 1786.3000,
                    21.4870, 21.4870,
                    14.5760, 14.5760,
                    1.2645, 1.2645,
                    1201.3000, 1201.3000,
                    0.0016, 0.0016
                ]

                baseline_input_feed = {
                    name: np.array([[value]], dtype=np.float64)
                    for name, value in zip(input_names, baseline_input_values)
                }

                baseline_prediction = session.run([output_name], baseline_input_feed)[0]
                baseline_yield = baseline_prediction[0][0]

                # Calculate difference
                yield_diff = predicted_yield - baseline_yield
                yield_diff_pct = (yield_diff / baseline_yield) * 100 if baseline_yield != 0 else 0

                # Display results
                st.markdown(f"""
                <div style="border: 2px solid #17a2b8; border-radius: 10px; padding: 15px; background-color: #e7f5f8;">
                    <h4 style="color: #17a2b8; margin:0;">Scenario: {scenario}</h4>
                    <p style="font-size: 1.8em; font-weight: bold; color: #0d6efd; margin:5px 0;">
                        Predicted Yield: {predicted_yield:.2f} g/L
                    </p>
                    <p style="font-size: 1.2em; color: {'#28a745' if yield_diff >= 0 else '#dc3545'}; margin:0;">
                        Change: {yield_diff:+.2f} g/L ({yield_diff_pct:+.1f}%)
                    </p>
                    <p style="font-size: 0.9em; color: #666; margin:5px 0;">
                        Baseline: {baseline_yield:.2f} g/L
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # Interpretation
                st.markdown("---")
                st.markdown("**📝 Interpretation:**")

                if "Metabolic Stress" in scenario:
                    st.warning("⚠️ **Metabolic Stress**: High activity (CER) but low growth (Biomass) indicates inefficiency, leading to **lower yield**.")
                elif "Metabolic Disconnect" in scenario:
                    st.error("🚨 **Metabolic Disconnect**: High Biomass with Low CER is biologically inconsistent (cells exist but aren't breathing). This indicates a severe anomaly or sensor fault.")
                elif "Process Failure" in scenario:
                    st.error("🚨 **Process Failure**: A drop in all key metabolic indicators (Biomass, CER, OUR) causes a **drastic yield collapse**.")
                elif "Sensor Failure" in scenario:
                    st.info("ℹ️ **Sensor Failure**: A massive spike in a low-impact feature (Substrate) has **almost no effect** on the prediction, demonstrating model robustness.")
                elif "Late Stage" in scenario:
                    st.info("📉 **Late Stage**: Typical end-of-batch signs (low substrate, slowing respiration) result in a **moderate yield decrease**.")
                elif "Biomass" in scenario or "Carbon" in scenario:
                    st.success("✅ Changes to **high-impact features** (Biomass, Carbon Evolution Rate) produce **significant** yield changes.")
                elif "Substrate" in scenario or "PAA" in scenario:
                    st.info("ℹ️ Changes to **low-impact features** (Substrate, PAA) produce **minimal** yield changes, despite high variability.")
                elif "Combined" in scenario:
                    st.success("✅ Combined high-impact drift shows **amplified effects** on yield prediction.")

            except Exception as e:
                st.error(f"🔴 Error in drift scenario: {e}")
        else:
            st.error("Model not loaded. Cannot run drift scenario.")

