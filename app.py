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
        st.success("✅ Model loaded successfully!")
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

# --- Header ---
st.title("💊 Penicillin Yield Predictor")
st.markdown("An interactive demo to predict penicillin yield using a pre-trained ONNX model. Adjust the sliders and see the prediction change in real-time!")

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

    **🟢 LOW IMPACT Features** (Weak predictors):
    - Oxygen Uptake Rate (correlation: 0.21)
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
    get_model_details(session)

    st.markdown("---")

    # --- Main Layout (Inputs and Outputs side-by-side) ---
    col1, col2 = st.columns([1, 1]) # Create two columns of equal width

    with col1:
        st.header("⚙️ Input Features")
        st.write("Use the controls below to set the penicillin yield features.")
        st.info("🔴 = High impact on prediction | 🟢 = Low impact on prediction")

        # ==============================================================================
        # TODO: CUSTOMIZE YOUR INPUTS HERE
        # This section is hardcoded for a better user experience.
        # You must replace these with the inputs your specific model requires.
        # ==============================================================================

        # Create two columns for the inputs to make the layout cleaner
        input_col1, input_col2 = st.columns(2)

        with input_col1:
            cer_0 = st.number_input("🔴 Carbon Evolution Rate (g/h) [0]", value=1.4006, format="%.4f",
                                    help="High impact feature (corr: 0.72) - Monitors microbial metabolic activity")
            nh3_0 = st.number_input("🟢 NH3 Concentration (g/L) [0]", value=1786.3000, format="%.4f",
                                    help="Low impact feature (corr: 0.06) - Nitrogen source tracking")
            biomass_0 = st.number_input("🔴 Offline Biomass (g/L) [0]", value=21.4870, format="%.4f",
                                        help="Highest impact feature (corr: 0.79) - Cell concentration measurement")
            penicillin_0 = st.number_input("Offline Penicillin (g/L) [0]", value=14.5760, format="%.4f",
                                           help="Historical yield measurement at time [0]")
            our_0 = st.number_input("🟢 Oxygen Uptake Rate (g/min) [0]", value=1.2645, format="%.4f",
                                    help="Low impact feature (corr: 0.21) - Respiration rate")
            paa_0 = st.number_input("🟢 PAA Concentration (g/L) [0]", value=1201.3000, format="%.4f",
                                    help="Low impact, high drift (CV: 83%) - Precursor compound")
            substrate_0 = st.number_input("🟢 Substrate Concentration (g/L) [0]", value=0.0016, format="%.4f",
                                          help="Low impact, extremely high drift (CV: 347%) - Nutrient level")

        with input_col2:
            cer_1 = st.number_input("🔴 Carbon Evolution Rate (g/h) [1]", value=1.4006, format="%.4f",
                                    help="High impact feature (corr: 0.72) - Monitors microbial metabolic activity")
            nh3_1 = st.number_input("🟢 NH3 Concentration (g/L) [1]", value=1786.3000, format="%.4f",
                                    help="Low impact feature (corr: 0.06) - Nitrogen source tracking")
            biomass_1 = st.number_input("🔴 Offline Biomass (g/L) [1]", value=21.4870, format="%.4f",
                                        help="Highest impact feature (corr: 0.79) - Cell concentration measurement")
            penicillin_1 = st.number_input("Offline Penicillin (g/L) [1]", value=14.5760, format="%.4f",
                                           help="Historical yield measurement at time [1]")
            our_1 = st.number_input("🟢 Oxygen Uptake Rate (g/min) [1]", value=1.2645, format="%.4f",
                                    help="Low impact feature (corr: 0.21) - Respiration rate")
            paa_1 = st.number_input("🟢 PAA Concentration (g/L) [1]", value=1201.3000, format="%.4f",
                                    help="Low impact, high drift (CV: 83%) - Precursor compound")
            substrate_1 = st.number_input("🟢 Substrate Concentration (g/L) [1]", value=0.0016, format="%.4f",
                                          help="Low impact, extremely high drift (CV: 347%) - Nutrient level")

        # The "Run Prediction" button is centered using columns
        _, btn_col, _ = st.columns([1, 2, 1])
        with btn_col:
            predict_button = st.button("⚡ Predict Yield", use_container_width=True)

    with col2:
        st.header("📈 Prediction Output")
        st.write("The model's predicted yield will appear below.")

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
            "PAA Drift (+100%)",
            "Combined High-Impact Drift"
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
        "PAA Drift (+100%)": {
            "biomass_mult": 1.0, "cer_mult": 1.0, "substrate_mult": 1.0,
            "paa_mult": 2.0, "our_mult": 1.0, "nh3_mult": 1.0
        },
        "Combined High-Impact Drift": {
            "biomass_mult": 1.5, "cer_mult": 1.5, "substrate_mult": 1.0,
            "paa_mult": 1.0, "our_mult": 1.0, "nh3_mult": 1.0
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

                if "Biomass" in scenario or "Carbon" in scenario:
                    st.success("✅ Changes to **high-impact features** (Biomass, Carbon Evolution Rate) produce **significant** yield changes.")
                elif "Substrate" in scenario or "PAA" in scenario:
                    st.info("ℹ️ Changes to **low-impact features** (Substrate, PAA) produce **minimal** yield changes, despite high variability.")
                elif "Combined" in scenario:
                    st.success("✅ Combined high-impact drift shows **amplified effects** on yield prediction.")

            except Exception as e:
                st.error(f"🔴 Error in drift scenario: {e}")
        else:
            st.error("Model not loaded. Cannot run drift scenario.")

