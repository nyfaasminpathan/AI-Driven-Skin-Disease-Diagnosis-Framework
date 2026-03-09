import streamlit as st
import base64

st.set_page_config(
    page_title="AI-Driven Skin Disease Diagnosis Framework",
    layout="wide"
)

# -------------------------
# State Management
# -------------------------
if 'show_uploader' not in st.session_state:
    st.session_state.show_uploader = False

# reset when coming back from analyze page
if 'uploaded_image' in st.session_state and not st.session_state.show_uploader:
    del st.session_state['uploaded_image']

# -------------------------
# Load Hero Image
# -------------------------
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img:
            return base64.b64encode(img.read()).decode()
    except:
        return ""

hero_img = get_base64_image("assets/hero.png")

# -------------------------
# CSS
# -------------------------
st.markdown("""
<style>

html{
scroll-behavior:smooth;
}

html, body, [class*="css"]{
font-family:'Segoe UI',sans-serif;
}

/* Hide Streamlit header */
header {visibility: hidden;}
[data-testid="stToolbar"] {display:none;}

a.header-anchor {display:none !important;}
.stMarkdown a svg {display:none !important;}

.block-container{
padding-top:0rem !important;
padding-left:0rem !important;
padding-right:0rem !important;
max-width:100% !important;
}

/* NAVBAR */
.navbar{
display:flex;
justify-content:space-between;
align-items:center;
padding:15px 60px;
background:linear-gradient(90deg,#0ea5a4,#14b8a6);
box-shadow:0 4px 15px rgba(0,0,0,0.1);
position:sticky;
top:0;
z-index:999;
}

.logo{
font-size:22px;
font-weight:bold;
color:white;
}

.nav-links a{
margin-left:20px;
text-decoration:none;
font-weight:600;
padding:8px 18px;
border-radius:20px;
transition:0.3s;
color:white;
background:rgba(255,255,255,0.15);
}

.nav-links a:hover{
background:white;
color:#0ea5a4;
transform:translateY(-2px);
box-shadow:0 4px 10px rgba(0,0,0,0.15);
}

/* HERO */
.hero-text h1{
font-size:48px;
color:#0f172a;
margin-top:40px;
}

.hero-text p{
font-size:20px;
color:#475569;
margin-bottom:25px;
margin-top:5px;
max-width:600px;
margin-left:auto;
margin-right:auto;
}

.hero-image{
text-align:center;
margin-top:10px;
margin-left:10px;
}

.hero-image img{
width:100%;
max-width:550px;
mix-blend-mode:multiply;
}

/* BUTTON */
.stButton button{
background:#0ea5a4;
color:white;
padding:12px 30px;
font-size:16px;
border-radius:8px;
border:none;
}

.stButton button:hover{
background:#0f766e;
}

/* HOW SECTION */
.how-heading{
text-align:center;
margin-bottom:40px;
font-size:36px;
font-weight:700;
color:#0ea5a4;
position:relative;
display:inline-block;
padding-bottom:10px;
}

.how-heading::after{
content:"";
display:block;
width:80px;
height:4px;
background:#0ea5a4;
margin:10px auto 0 auto;
border-radius:10px;
}

/* CARDS */
.cards-container{
padding:0 60px;
}

.card{
padding:30px;
border-radius:15px;
text-align:center;
transition:0.3s;
height:100%;
}

.card1{
background:#5eead4;
}

.card2{
background:#7dd3fc;
}

.card3{
background:#fcd34d;
}

.card:hover{
transform:translateY(-8px);
box-shadow:0 8px 25px rgba(0,0,0,0.1);
}

/* ABOUT */
.about-container{
padding:0 60px;
margin-bottom:60px;
}

.about{
margin-top:60px;
padding:70px;
background:linear-gradient(to right,#0ea5a4,#14b8a6);
border-radius:15px;
text-align:center;
color:white;
}

.about h2{
font-size:36px;
margin-bottom:20px;
}

.about p{
max-width:800px;
margin:15px auto;
font-size:18px;
line-height:1.6;
}

</style>
""", unsafe_allow_html=True)

# -------------------------
# NAVBAR
# -------------------------
st.markdown("""
<div class="navbar">
<div class="logo">AI-Driven Skin Disease Diagnosis Framework</div>
<div class="nav-links">
<a href="#home">Home</a>
<a href="#how">How It Works</a>
<a href="#about">About</a>
</div>
</div>
""", unsafe_allow_html=True)

# =================================================
# HERO SECTION
# =================================================
st.markdown('<div id="home">', unsafe_allow_html=True)

col_space1, col1, col2, col_space2 = st.columns([0.5,4,3,0.5])

with col1:

    st.markdown(f"""
    <div class="hero-text" style="text-align:center;">
        <h1>AI-Powered Skin Disease Detection</h1>
        <p>
        Upload a skin image and let our AI system analyze visual patterns
        to classify bacterial, fungal, or viral skin conditions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_center, col_right = st.columns([1,2,1])

    with col_center:

        if not st.session_state.show_uploader:

            if st.button("Start Skin Analysis"):
                st.session_state.show_uploader = True
                st.rerun()

    if st.session_state.show_uploader:

        st.markdown('<div style="text-align:center; margin-top:20px;">', unsafe_allow_html=True)

        st.write("Drag and drop a clear, well-lit image.")
        uploaded_file = st.file_uploader("", type=["jpg","jpeg","png"])

        if uploaded_file is not None:

            st.session_state['uploaded_image'] = uploaded_file
            st.switch_page("pages/analyze.py")

        if st.button("Cancel"):
            st.session_state.show_uploader = False
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

with col2:

    st.markdown(f"""
    <div class="hero-image">
        <img src="data:image/png;base64,{hero_img}">
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# =================================================
# HOW IT WORKS
# =================================================
st.markdown('<div id="how">', unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;">
<h2 class="how-heading">How It Works</h2>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="cards-container">', unsafe_allow_html=True)

col1,col2,col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="card card1">
    <h3>Upload Image</h3>
    <p>Upload a clear photo of the affected skin area.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card card2">
    <h3>AI Analysis</h3>
    <p>Deep learning model analyzes patterns and textures.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card card3">
    <h3>Get Results</h3>
    <p>Receive an instant analysis of your skin condition.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# =================================================
# ABOUT
# =================================================
st.markdown("""
<div id="about" class="about-container">
<div class="about">
<h2>About</h2>

<p>
Skin diseases are common health conditions that affect people of all ages. 
They range from minor infections to more serious conditions and can be caused 
by various factors such as bacteria, viruses, and fungi. Early detection plays 
an important role in understanding and managing these conditions.
</p>

<p>
This website helps users gain preliminary insights into possible skin 
conditions by allowing them to upload images of affected skin areas. 
It aims to provide a simple and accessible way to understand potential 
skin issues.
</p>

</div>
</div>
""", unsafe_allow_html=True)