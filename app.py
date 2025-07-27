# import streamlit as st
# import pandas as pd
# from mplsoccer import Pitch
# import matplotlib.pyplot as plt

# # Page settings
# st.set_page_config(page_title="Football Pitch xG Visualizer", layout="wide")
# st.title("⚽ xG / Shot Visualizer on Football Pitch")

# # File uploader
# uploaded_file = st.file_uploader("📁 Upload your `.pkl` file (must include 'location' column)", type=["pkl"])

# if uploaded_file is not None:
#     try:
#         # Load DataFrame
#         df = pd.read_pickle(uploaded_file)
#         st.success("✅ File loaded successfully!")

#         # Show preview
#         st.subheader("📋 Data Preview")
#         st.dataframe(df.head())

#         if 'location' not in df.columns:
#             st.error("Missing required 'location' column.")
#             st.stop()

#         # Extract x and y from 'location' list
#         df = df[df['location'].notna()]
#         df['x'] = df['location'].apply(lambda loc: loc[0] if isinstance(loc, list) and len(loc) > 1 else None)
#         df['y'] = df['location'].apply(lambda loc: loc[1] if isinstance(loc, list) and len(loc) > 1 else None)
#         df = df.dropna(subset=['x', 'y'])

#         # Prediction column selection
#         pred_cols = [col for col in df.columns if col.startswith("pred")]
#         pred_col = st.selectbox("🎯 Choose prediction column (optional)", pred_cols + ["None"])

#         # Create pitch
#         pitch = Pitch(pitch_type='statsbomb', line_zorder=2)
#         fig, ax = pitch.draw(figsize=(10, 7))

#         # Scatter plot based on prediction
#         if pred_col != "None":
#             size_factor = st.slider("🌀 Scale prediction size", 100, 3000, 1000)
#             color_map = st.selectbox("🌈 Choose color map", ['viridis', 'plasma', 'coolwarm', 'magma', 'inferno'])
#             scatter = pitch.scatter(df['x'], df['y'],
#                                      s=df[pred_col]*size_factor,
#                                      c=df[pred_col],
#                                      cmap=color_map,
#                                      edgecolors='black',
#                                      alpha=0.7,
#                                      ax=ax)
#             fig.colorbar(scatter, ax=ax).set_label(f"{pred_col}")
#         else:
#             pitch.scatter(df['x'], df['y'], s=100, color='red', edgecolors='black', alpha=0.6, ax=ax)

#         st.pyplot(fig)

#     except Exception as e:
#         st.error(f"❌ Error loading or processing file: {e}")

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="PKL Visualizer", layout="wide")
st.title("📊 PKL File Viewer & Visualizer")

uploaded_file = st.file_uploader("📁 Upload a `.pkl` file", type=["pkl"])

if uploaded_file is not None:
    try:
        df = pd.read_pickle(uploaded_file)
        
        st.subheader("🧾 Data Preview")
        st.dataframe(df)

        st.subheader("📌 Data Info")
        st.text(f"Shape: {df.shape}")
        st.write(df.describe())

        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        if not numeric_cols:
            st.warning("No numeric columns found in the dataset.")
            st.stop()

        plot_type = st.selectbox("📈 Choose plot type", ["Histogram", "Scatter Plot", "Box Plot", "Correlation Heatmap", "Target vs Prediction"])

        if plot_type == "Histogram":
            column = st.selectbox("Choose column", numeric_cols)
            bins = st.slider("Bins", 5, 100, 30)
            fig, ax = plt.subplots()
            ax.hist(df[column], bins=bins, color="skyblue", edgecolor="black")
            ax.set_title(f"Histogram of {column}")
            st.pyplot(fig)

        elif plot_type == "Scatter Plot":
            x_col = st.selectbox("X-axis", numeric_cols, key="scatter_x")
            y_col = st.selectbox("Y-axis", numeric_cols, key="scatter_y")
            fig, ax = plt.subplots()
            ax.scatter(df[x_col], df[y_col], alpha=0.5, color='teal')
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f"Scatter Plot: {x_col} vs {y_col}")
            st.pyplot(fig)

        elif plot_type == "Box Plot":
            column = st.selectbox("Choose column", numeric_cols, key="box")
            fig, ax = plt.subplots()
            sns.boxplot(y=df[column], ax=ax, color='lightcoral')
            ax.set_title(f"Box Plot of {column}")
            st.pyplot(fig)

        elif plot_type == "Correlation Heatmap":
            fig, ax = plt.subplots(figsize=(10, 6))
            corr = df[numeric_cols].corr()
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
            ax.set_title("Correlation Heatmap")
            st.pyplot(fig)

        elif plot_type == "Target vs Prediction":
            target_col = st.selectbox("Target column", [col for col in df.columns if "target" in col.lower()])
            pred_cols = [col for col in df.columns if "pred" in col.lower()]
            if pred_cols:
                pred_col = st.selectbox("Prediction column", pred_cols)
                fig, ax = plt.subplots()
                sns.histplot(df, x=pred_col, hue=target_col, bins=30, palette="Set2", ax=ax, multiple="stack")
                ax.set_title(f"{pred_col} Distribution by {target_col}")
                st.pyplot(fig)
            else:
                st.warning("No prediction column found (e.g., starting with 'pred_').")

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
