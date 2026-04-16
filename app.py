from __future__ import annotations

import torch
from ultralytics.nn.tasks import DetectionModel

# 1. Trust the YOLO model structure immediately
torch.serialization.add_safe_globals([DetectionModel])

import time
from datetime import datetime
from typing import Optional, Dict, Any
import cv2
import streamlit as st
import numpy as np

# 2. SET PAGE CONFIG (Must be the first Streamlit command)
st.set_page_config(
    page_title="Mediseena: Vision-Based Ingestion System",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 3. NOW import your custom modules
from config import AppConfig
from detectors import DetectionPipeline
from display import DisplayRenderer
from logger import TrialLogger
from utils import distance_between, enhance_low_light
from verifier import FrameInputs, IngestionVerifier
# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================

CUSTOM_CSS = """<style>
:root{--primary-blue:#2563eb;--secondary-blue:#3b82f6;--success-green:#10b981;--warning-yellow:#f59e0b;--error-red:#ef4444;--slate-900:#0f172a;--slate-800:#1e293b;--slate-700:#334155;--slate-600:#475569;--slate-200:#e2e8f0;--slate-100:#f1f5f9}
.main{background-color:#f8fafc}
.main-header{background:linear-gradient(135deg,#2563eb 0%,#3b82f6 100%);padding:2rem;border-radius:12px;color:white;margin-bottom:2rem;box-shadow:0 4px 6px rgba(0,0,0,0.1)}
.main-header h1{color:white!important;font-size:2.5rem;font-weight:700;margin:0}
.main-header p{color:rgba(255,255,255,0.9);font-size:1.1rem;margin:0.5rem 0 0 0}
.med-card{background:linear-gradient(135deg,#eff6ff 0%,#dbeafe 100%);border:2px solid #3b82f6;border-radius:12px;padding:1.5rem;margin-bottom:1.5rem}
.med-card h3{color:#1e40af;font-size:1.5rem;font-weight:600;margin-bottom:0.5rem}
.status-badge{display:inline-block;padding:0.5rem 1rem;border-radius:9999px;font-weight:600;font-size:0.875rem;margin-top:0.5rem}
.status-verified{background-color:#10b981;color:white}
.status-pending{background-color:#64748b;color:white}
.state-circle{width:3rem;height:3rem;border-radius:50%;display:inline-flex;align-items:center;justify-content:center;font-weight:700;margin-bottom:0.5rem;border:3px solid #e2e8f0;background:white;color:#94a3b8}
.state-circle.completed{background:#10b981;border-color:#10b981;color:white}
.state-circle.active{background:#3b82f6;border-color:#3b82f6;color:white;animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.7}}
.state-label{font-size:0.875rem;color:#475569;font-weight:500}
.state-label.active{color:#1e40af;font-weight:600}
.stat-container{background:white;border-radius:8px;padding:1rem;border-left:4px solid #3b82f6;margin-bottom:1rem}
.stat-label{font-size:0.875rem;color:#64748b;margin-bottom:0.25rem}
.stat-value{font-size:1.5rem;font-weight:700;color:#0f172a}
.stat-value.success{color:#10b981}
.stat-value.primary{color:#3b82f6}
.instruction-step{display:flex;gap:1rem;margin-bottom:1rem;padding:1rem;background:#f8fafc;border-radius:8px}
.step-number{width:2rem;height:2rem;background:#3b82f6;color:white;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:700;flex-shrink:0}
.step-number.success{background:#10b981}
.detection-info{background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;padding:0.75rem;margin-bottom:0.5rem}
.detection-row{display:flex;justify-content:space-between;margin-bottom:0.5rem}
.detection-label{color:#475569;font-size:0.875rem}
.detection-value{font-weight:600;font-size:0.875rem}
.detection-value.positive{color:#10b981}
.detection-value.negative{color:#ef4444}
.log-entry{background:white;border-left:3px solid #10b981;padding:1rem;margin-bottom:0.5rem;border-radius:4px}
.log-entry.failed{border-left-color:#ef4444}
.log-time{color:#64748b;font-size:0.875rem}
.log-medication{color:#0f172a;font-weight:600;margin:0.25rem 0}
.stButton>button{background:linear-gradient(135deg,#3b82f6 0%,#2563eb 100%);color:white;border:none;border-radius:8px;padding:0.75rem 1.5rem;font-weight:600;transition:all 0.2s}
.stButton>button:hover{transform:translateY(-2px);box-shadow:0 4px 12px rgba(37,99,235,0.3)}
</style>"""

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def init_session_state():
    """Initialize Streamlit session state variables"""
    if 'initialized' not in st.session_state:
        st.session_state.config = AppConfig()
        st.session_state.detector = DetectionPipeline(st.session_state.config)
        st.session_state.logger = TrialLogger(st.session_state.config.csv_path)
        st.session_state.verifier = IngestionVerifier(st.session_state.config, st.session_state.logger)
        st.session_state.renderer = DisplayRenderer(st.session_state.config)
        st.session_state.cap = None
        st.session_state.camera_active = False
        st.session_state.current_state = "WAITING"
        st.session_state.last_verified_time = None
        st.session_state.frame_counter = 0
        
        # Mock data
        st.session_state.medications = [
            {"id": 1, "name": "Lisinopril", "dosage": "10mg", "frequency": "Once daily", "time": "08:00 AM", "color": "blue", "instructions": "Take with food"},
            {"id": 2, "name": "Metformin", "dosage": "500mg", "frequency": "Twice daily", "time": "08:00 AM, 08:00 PM", "color": "yellow", "instructions": "Take with meals"},
            {"id": 3, "name": "Atorvastatin", "dosage": "20mg", "frequency": "Once daily", "time": "08:00 PM", "color": "green", "instructions": "Take at bedtime"}
        ]
        
        st.session_state.adherence = {
            "daily_rate": 95.2, "weekly_rate": 93.8, "monthly_rate": 94.5,
            "streak": 12, "total_doses": 84, "verified_doses": 80, "missed_doses": 4
        }
        
        st.session_state.verification_logs = []
        st.session_state.initialized = True

# ============================================================================
# CAMERA FUNCTIONS
# ============================================================================

def start_camera():
    """Start camera capture - exactly like main.py"""
    if st.session_state.cap is None:
        st.session_state.cap = cv2.VideoCapture(st.session_state.config.camera_index)
        
        if not st.session_state.cap.isOpened():
            st.error(f"❌ Unable to open camera index {st.session_state.config.camera_index}")
            return False
        
        st.session_state.camera_active = True
        return True
    return True

def stop_camera():
    """Stop camera capture"""
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None
    st.session_state.camera_active = False

def enhance_frame_quality(frame):
    """Minimal enhancement - preserve natural colors"""
    # No enhancement - return raw frame for true colors
    return frame

def process_frame():
    """Process single frame - EXACTLY like main.py"""
    if st.session_state.cap is None or not st.session_state.cap.isOpened():
        return None, None

    ret, frame = st.session_state.cap.read()
    if not ret or frame is None:
        return None, None

    current_time = time.time()
    
    # EXACTLY like main.py: enhance low light first
    frame = enhance_low_light(frame)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run detection pipeline (same as main.py)
    hand = st.session_state.detector.detect_hand(rgb, frame.shape)
    pill = st.session_state.detector.detect_pill(frame, hand.box)
    mouth = st.session_state.detector.detect_mouth(rgb, frame.shape)
    distance_to_mouth = distance_between(hand.center, mouth.center)

    # Update verifier
    session = st.session_state.verifier.update(
        FrameInputs(
            pill_detected=pill.box is not None,
            pill_in_hand=pill.in_hand,
            hand_visible=hand.box is not None,
            mouth_open=mouth.is_open,
            distance_to_mouth=distance_to_mouth,
        ),
        current_time,
    )

    st.session_state.current_state = session.state

    # Check for verification
    if session.state == "VERIFIED" and st.session_state.last_verified_time != current_time:
        st.session_state.last_verified_time = current_time
        add_verification_log("verified")

    # Render AR overlays - EXACTLY like main.py
    frame = st.session_state.renderer.draw(
        frame, session, current_time=current_time,
        hand_box=hand.box, pill_box=pill.box, pill_conf=pill.confidence,
        mouth_center=mouth.center, pill_in_hand_flag=pill.in_hand,
    )

    detection_data = {
        "pill_detected": pill.box is not None,
        "pill_in_hand": pill.in_hand,
        "hand_visible": hand.box is not None,
        "mouth_open": mouth.is_open,
        "distance_to_mouth": distance_to_mouth,
        "pill_confidence": pill.confidence if pill.confidence else 0.0,
    }

    # Store last detection data in session state
    st.session_state.last_detection_data = detection_data

    # Return BGR frame (will be converted to RGB for Streamlit display)
    return frame, detection_data

def add_verification_log(status: str):
    """Add verification to log"""
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "medication": f"{st.session_state.medications[0]['name']} {st.session_state.medications[0]['dosage']}",
        "status": status,
        "notes": "Automated verification via computer vision"
    }
    st.session_state.verification_logs.insert(0, log_entry)
    if status == "verified":
        st.session_state.adherence["verified_doses"] += 1
        st.session_state.adherence["total_doses"] += 1

# ============================================================================
# UI HELPER FUNCTIONS
# ============================================================================

def html_card(content: str, class_name: str = "med-card") -> str:
    """Generate HTML card"""
    return f'<div class="{class_name}">{content}</div>'

def html_stat(label: str, value: str, value_class: str = "") -> str:
    """Generate stat container HTML"""
    return f'''<div class="stat-container">
        <div class="stat-label">{label}</div>
        <div class="stat-value {value_class}">{value}</div>
    </div>'''

# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_header():
    """Render main header"""
    st.markdown('''<div class="main-header">
        <h1>💊 Mediseena: Vision-Based Ingestion System </h1>
        <p>Real-time vision-based ingestion monitoring powered by AI</p>
    </div>''', unsafe_allow_html=True)

def render_current_medication():
    """Render current medication card"""
    med = st.session_state.medications[0]
    status_class = "status-verified" if st.session_state.current_state == "VERIFIED" else "status-pending"
    status_text = "✓ Verified" if st.session_state.current_state == "VERIFIED" else "Pending Verification"
    
    st.markdown(f'''<div class="med-card">
        <div style="display:flex;justify-content:space-between;align-items:start;">
            <div>
                <h3>💊 {med['name']} {med['dosage']}</h3>
                <p style="color:#475569;margin:0.5rem 0;">🕐 {med['time']} • 📅 {datetime.now().strftime("%B %d, %Y")}</p>
                <p style="color:#1e40af;margin:0.5rem 0;font-size:0.9rem;">ℹ️ {med['instructions']}</p>
            </div>
            <span class="status-badge {status_class}">{status_text}</span>
        </div>
    </div>''', unsafe_allow_html=True)

def render_state_stepper_dynamic(placeholder, detection_data=None):
    """Render FSM state progression stepper dynamically with real-time detection feedback"""
    states = [
        {"name": "WAITING", "label": "Waiting for Pill", "icon": "⏳"},
        {"name": "PILL_IN_HAND", "label": "Pill in Hand", "icon": "✋"},
        {"name": "HAND_NEAR_MOUTH", "label": "Hand Near Mouth", "icon": "👄"},
        {"name": "VERIFIED", "label": "Ingestion Verified", "icon": "✅"}
    ]
    
    state_order = ["WAITING", "PILL_IN_HAND", "HAND_NEAR_MOUTH", "VERIFIED"]
    current_index = state_order.index(st.session_state.current_state) if st.session_state.current_state in state_order else 0
    
    # Build HTML for all states with detection feedback
    stepper_html = '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin-bottom:2rem;">'
    
    for i, state in enumerate(states):
        circle_class = "completed" if i < current_index else ("active" if i == current_index else "")
        label_class = "active" if i == current_index else ""
        icon = "✓" if i < current_index else (state["icon"] if i == current_index else str(i + 1))
        
        # Add detection status feedback for active step
        detection_hint = ""
        if i == current_index and detection_data:
            if state["name"] == "WAITING":
                if detection_data.get("pill_detected"):
                    detection_hint = '<div style="font-size:0.7rem;color:#10b981;margin-top:0.25rem;">✓ Pill detected!</div>'
                else:
                    detection_hint = '<div style="font-size:0.7rem;color:#64748b;margin-top:0.25rem;">Looking for pill...</div>'
            elif state["name"] == "PILL_IN_HAND":
                if detection_data.get("pill_in_hand"):
                    detection_hint = '<div style="font-size:0.7rem;color:#10b981;margin-top:0.25rem;">✓ Pill in hand!</div>'
                elif detection_data.get("pill_detected"):
                    detection_hint = '<div style="font-size:0.7rem;color:#f59e0b;margin-top:0.25rem;">Move to hand...</div>'
            elif state["name"] == "HAND_NEAR_MOUTH":
                dist = detection_data.get("distance_to_mouth", 999)
                threshold = st.session_state.config.approach_dist
                if dist < threshold:
                    detection_hint = '<div style="font-size:0.7rem;color:#10b981;margin-top:0.25rem;">✓ Near mouth!</div>'
                else:
                    detection_hint = f'<div style="font-size:0.7rem;color:#f59e0b;margin-top:0.25rem;">Distance: {dist:.0f}px (need <{threshold}px)</div>'
        
        stepper_html += f'''<div class="state-step" style="text-align:center;">
            <div class="state-circle {circle_class}">{icon}</div>
            <div class="state-label {label_class}">{state['label']}{detection_hint}</div>
        </div>'''
    
    stepper_html += '</div>'
    placeholder.markdown(stepper_html, unsafe_allow_html=True)

def render_camera_feed_with_state_updates(state_stepper_placeholder):
    """Render live camera feed with real-time state updates - smooth and flicker-free like main.py"""
    st.markdown("### 📹 Live Camera Feed")
    
    if st.session_state.camera_active:
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("⏹️ Stop Camera", key="stop_camera"):
                stop_camera()
                st.rerun()
        
        # Persistent placeholders for smooth updates
        camera_placeholder = st.empty()
        status_placeholder = st.empty()
        
        # Debug checkbox to show raw vs processed
        show_raw = st.checkbox("Show Raw Camera (Debug)", value=False, key="show_raw_debug")
        
        # Continuous loop - similar to main.py approach
        # Process many frames before page naturally needs to refresh
        max_frames = 300  # ~10 seconds at 30fps before auto-refresh
        frame_count = 0
        
        while frame_count < max_frames and st.session_state.camera_active:
            try:
                if show_raw:
                    # Show RAW camera feed without any processing
                    ret, frame = st.session_state.cap.read()
                    if ret and frame is not None:
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        camera_placeholder.image(rgb_frame, channels="RGB", use_container_width=True, caption="RAW Camera Feed")
                        frame_count += 1
                        time.sleep(0.033)
                    else:
                        break
                else:
                    # Show processed feed with overlays
                    annotated_frame, detection_data = process_frame()
                    
                    if annotated_frame is not None:
                        # Convert BGR to RGB for proper color display
                        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                        
                        # Update display smoothly (no page reload)
                        camera_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
                        
                        # Update state stepper every 10 frames for real-time feedback
                        if frame_count % 10 == 0:
                            render_state_stepper_dynamic(state_stepper_placeholder, detection_data)
                        
                        frame_count += 1
                        st.session_state.frame_counter += 1
                        
                        # Small delay for smooth 30fps playback
                        time.sleep(0.033)
                    else:
                        camera_placeholder.markdown('''<div style="padding:1rem;background:#fee;border-radius:8px;color:#c00;">
                            ⚠️ Camera disconnected. Check connection and restart.
                        </div>''', unsafe_allow_html=True)
                        break
                    
            except Exception as e:
                camera_placeholder.error(f"⚠️ Camera error: {str(e)}")
                break
        
        # Show live indicator
        status_placeholder.markdown('''<div style="text-align:right;margin-top:0.5rem;">
            <span style="background:#ef4444;color:white;padding:0.5rem 1rem;border-radius:9999px;font-weight:600;font-size:0.875rem;">🔴 LIVE</span>
        </div>''', unsafe_allow_html=True)
        
        # Only rerun after processing full batch (smooth, no flicker)
        if st.session_state.camera_active and frame_count >= max_frames:
            st.rerun()
            
    else:
        st.markdown('''<div style="background:#1e293b;border-radius:12px;padding:4rem 2rem;text-align:center;">
            <div style="font-size:4rem;margin-bottom:1rem;">📷</div>
            <p style="color:#94a3b8;font-size:1.1rem;margin-bottom:1.5rem;">Camera is off</p>
        </div>''', unsafe_allow_html=True)
        
        if st.button("▶️ Start Camera", key="start_camera", use_container_width=True):
            if start_camera():
                st.session_state.frame_counter = 0
                st.rerun()

def render_instructions():
    """Render instruction steps"""
    st.markdown("### 📋 Instructions")
    instructions = [
        ("1", "Position yourself", "Sit in front of the camera with good lighting", False),
        ("2", "Hold the pill", "Show the medication to the camera", False),
        ("3", "Take the medication", "Bring your hand to your mouth and swallow", False),
        ("✓", "Verification complete", "System confirms ingestion", True)
    ]
    
    for num, title, desc, is_success in instructions:
        step_class = "success" if is_success else ""
        st.markdown(f'''<div class="instruction-step">
            <div class="step-number {step_class}">{num}</div>
            <div>
                <div style="font-weight:600;color:#0f172a;margin-bottom:0.25rem;">{title}</div>
                <div style="color:#64748b;font-size:0.875rem;">{desc}</div>
            </div>
        </div>''', unsafe_allow_html=True)

def render_today_progress():
    """Render today's progress stats"""
    st.markdown("### 📊 Today's Progress")
    total, taken = 3, 2
    progress = (taken / total) * 100
    
    st.markdown(f'''
        {html_stat("Doses Taken", f"{taken}/{total}")}
        <div class="progress-bar" style="margin-top:0.5rem;margin-bottom:1rem;">
            <div class="progress-fill" style="width:{progress}%;"></div>
        </div>
        {html_stat("Next Dose", "2:00 PM", "primary")}
        {html_stat("Weekly Adherence", f"{st.session_state.adherence['weekly_rate']:.1f}%", "success")}
        {html_stat("Current Streak", f"{st.session_state.adherence['streak']} days", "primary")}
    ''', unsafe_allow_html=True)

def render_detection_details():
    """Render real-time detection details"""
    st.markdown("### 🔍 Detection Details")
    
    if st.session_state.camera_active and 'last_detection_data' in st.session_state:
        detection_data = st.session_state.last_detection_data
        if detection_data:
            # Add debug information for troubleshooting
            dist = detection_data.get('distance_to_mouth', 999)
            threshold = st.session_state.config.approach_dist
            
            rows = [
                ("Hand Visible", detection_data['hand_visible']),
                ("Pill Detected", detection_data['pill_detected']),
                ("Pill in Hand", detection_data['pill_in_hand']),
                ("Mouth Open", detection_data.get('mouth_open', False)),
            ]
            
            det_html = '<div class="detection-info">'
            for label, val in rows:
                val_class = 'positive' if val else 'negative'
                val_text = 'Yes ✓' if val else 'No ✗'
                det_html += f'<div class="detection-row"><span class="detection-label">{label}:</span><span class="detection-value {val_class}">{val_text}</span></div>'
            
            det_html += f'<div class="detection-row"><span class="detection-label">Distance to Mouth:</span><span class="detection-value">{dist:.1f}px</span></div>'
            det_html += f'<div class="detection-row"><span class="detection-label">Threshold:</span><span class="detection-value">{threshold}px</span></div>'
            
            # Add status indicator
            if dist < threshold:
                det_html += f'<div class="detection-row"><span class="detection-label">Status:</span><span class="detection-value positive">Within Range! ✓</span></div>'
            else:
                det_html += f'<div class="detection-row"><span class="detection-label">Status:</span><span class="detection-value negative">Too Far ({dist - threshold:.0f}px over)</span></div>'
            
            det_html += f'<div class="detection-row"><span class="detection-label">Pill Confidence:</span><span class="detection-value">{detection_data["pill_confidence"]*100:.1f}%</span></div>'
            det_html += '</div>'
            
            st.markdown(det_html, unsafe_allow_html=True)
            
            # Add console-style debug log
            st.markdown("#### 📊 Live Debug Log")
            debug_text = f"""
Current State: {st.session_state.current_state}
Distance: {dist:.1f}px (threshold: {threshold}px)
Hand Visible: {detection_data['hand_visible']}
Mouth Open: {detection_data.get('mouth_open', False)}
Pill in Hand: {detection_data['pill_in_hand']}

Status Check:
- Distance OK? {dist < threshold} (need < {threshold}px)
- Hand visible? {detection_data['hand_visible']}
- Mouth open? {detection_data.get('mouth_open', False)}
"""
            st.code(debug_text, language="text")
    else:
        st.info("Start camera to see live detection data")

# ============================================================================
# SIDEBAR
# ============================================================================

def render_sidebar():
    """Render sidebar navigation and controls"""
    with st.sidebar:
        st.markdown("## 🏥 Mediseena")
        st.markdown("---")
        
        page = st.radio("Navigation", 
            ["🏠 Verification", "📈 Adherence", "📋 Logs", "💊 Schedule", "⚙️ Settings"],
            key="navigation")
        
        st.markdown("---")
        st.markdown("### 🔧 System Status")
        camera_status = "🟢 Active" if st.session_state.camera_active else "🔴 Inactive"
        st.markdown(f"**Camera:** {camera_status}")
        st.markdown(f"**Current State:** {st.session_state.current_state}")
        
        st.markdown("---")
        st.markdown("### ⚡ Quick Actions")
        
        if st.button("🔄 Reset Verification", use_container_width=True):
            st.session_state.verifier = IngestionVerifier(st.session_state.config, st.session_state.logger)
            st.session_state.current_state = "WAITING"
            st.success("Verification reset!")
            st.rerun()
        
        if st.button("🆘 Help / Emergency", use_container_width=True):
            st.warning("Emergency assistance requested!")
        
        st.markdown("---")
        st.markdown(f'<div style="font-size:0.75rem;color:#64748b;text-align:center;">Version 1.0.0<br>{datetime.now().strftime("%B %d, %Y")}</div>', unsafe_allow_html=True)
        
        return page

# ============================================================================
# PAGES
# ============================================================================

def page_verification():
    """Main verification page with real-time state updates"""
    render_header()
    render_current_medication()
    st.markdown("---")
    st.markdown("### 📍 Verification Progress")
    
    # Create persistent placeholder for state stepper that updates in real-time
    state_stepper_placeholder = st.empty()
    render_state_stepper_dynamic(state_stepper_placeholder)
    
    st.markdown("---")
    
    col_left, col_right = st.columns([2, 1])
    with col_left:
        render_camera_feed_with_state_updates(state_stepper_placeholder)
    with col_right:
        render_instructions()
        st.markdown("---")
        render_today_progress()
        if st.session_state.camera_active:
            st.markdown("---")
            render_detection_details()

def page_adherence():
    """Adherence tracking page"""
    render_header()
    st.markdown("## 📈 Adherence Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        (col1, "Daily Rate", f"{st.session_state.adherence['daily_rate']:.1f}%", "2.1%"),
        (col2, "Weekly Rate", f"{st.session_state.adherence['weekly_rate']:.1f}%", "1.5%"),
        (col3, "Current Streak", f"{st.session_state.adherence['streak']} days", "3 days"),
        (col4, "Total Verified", st.session_state.adherence['verified_doses'], f"{st.session_state.adherence['missed_doses']} missed")
    ]
    
    for col, label, value, delta in metrics:
        with col:
            st.metric(label=label, value=value, delta=delta)
    
    st.markdown("---")
    st.markdown("### 📅 Calendar View")
    st.info("Calendar view with daily compliance would be displayed here")

def page_logs():
    """Verification logs page"""
    render_header()
    st.markdown("## 📋 Verification Logs")
    
    if st.session_state.verification_logs:
        for log in st.session_state.verification_logs[:20]:
            status_icon = "✓" if log['status'] == "verified" else "✗"
            st.markdown(f'''<div class="log-entry">
                <div class="log-time">{log['timestamp']}</div>
                <div class="log-medication">{status_icon} {log['medication']}</div>
                <div style="color:#64748b;font-size:0.875rem;">{log['notes']}</div>
            </div>''', unsafe_allow_html=True)
    else:
        st.info("No verification logs yet. Complete a verification to see logs here.")

def page_schedule():
    """Medication schedule page"""
    render_header()
    st.markdown("## 💊 Medication Schedule")
    
    for med in st.session_state.medications:
        st.markdown(f'''<div class="med-card">
            <h3>{med['name']} {med['dosage']}</h3>
            <p style="color:#475569;margin:0.5rem 0;">🕐 {med['time']} • {med['frequency']}</p>
            <p style="color:#1e40af;margin:0.5rem 0;font-size:0.9rem;">ℹ️ {med['instructions']}</p>
        </div>''', unsafe_allow_html=True)

def page_settings():
    """Settings page"""
    render_header()
    st.markdown("## ⚙️ Settings")
    st.markdown("### Camera Settings")
    
    camera_index = st.number_input("Camera Index", min_value=0, max_value=10, 
                                   value=st.session_state.config.camera_index)
    
    if camera_index != st.session_state.config.camera_index:
        st.session_state.config.camera_index = camera_index
        if st.session_state.camera_active:
            stop_camera()
            st.info("Camera settings updated. Please restart camera.")
    
    st.markdown("---")
    st.markdown("### System Information")
    st.info(f"""
        - **Version:** 1.0.0
        - **Camera Index:** {st.session_state.config.camera_index}
        - **CSV Log Path:** {st.session_state.config.csv_path}
    """)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point"""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    init_session_state()
    
    page = render_sidebar()
    
    # Page routing
    pages = {
        "🏠 Verification": page_verification,
        "📈 Adherence": page_adherence,
        "📋 Logs": page_logs,
        "💊 Schedule": page_schedule,
        "⚙️ Settings": page_settings
    }
    
    pages[page]()
    
    # Cleanup
    if not st.session_state.camera_active and st.session_state.cap is not None:
        stop_camera()

if __name__ == "__main__":
    main()