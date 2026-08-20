import serial
import serial.tools.list_ports
import threading
from collections import deque
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

# ==========================================
# 1. Configuration & 4000-Sample Buffers
# ==========================================
BUFFER_SIZE = 2000 
BAUD_RATE = 115200

# Two independent rolling buffers
ecg_buffer = deque([0] * BUFFER_SIZE, maxlen=BUFFER_SIZE)
pcg_buffer = deque([0] * BUFFER_SIZE, maxlen=BUFFER_SIZE)

def find_esp32_port():
    for port in serial.tools.list_ports.comports():
        if "CP210" in port.description or "CH340" in port.description or "USB Serial" in port.description:
            return port.device
    return None

# ==========================================
# 2. Background Serial Reader
# ==========================================
def read_serial():
    port_name = find_esp32_port()
    if not port_name:
        print("ESP32 not found. Check your USB connection.")
        return

    try:
        ser = serial.Serial(port_name, BAUD_RATE, timeout=1)
        print(f"Connected to {port_name}. Buffering Dual Signals...")
        
        while True:
            try:
                line = ser.readline().decode('utf-8').strip()
                # Ensure the line contains our comma separator
                if "," in line:
                    ecg_str, pcg_str = line.split(",")
                    ecg_buffer.append(int(ecg_str))
                    pcg_buffer.append(int(pcg_str))
            except Exception:
                pass # Ignore serial collisions
    except Exception as e:
        print(f"Serial connection failed: {e}")

threading.Thread(target=read_serial, daemon=True).start()

# ==========================================
# 3. Web Dashboard (Side-by-Side Layout)
# ==========================================
app = dash.Dash(__name__)

app.layout = html.Div(style={'font-family': 'Arial'}, children=[
    html.H1("Dual Biosignal Acquisition Dashboard"),
    html.P(f"Displaying the latest {BUFFER_SIZE} samples per channel (Auto-scaled)."),
    
    # CSS Flexbox container to put graphs side-by-side
    html.Div(style={'display': 'flex', 'flex-direction': 'row', 'width': '100%'}, children=[
        
        # Left Panel: ECG
        html.Div(style={'width': '50%', 'padding': '10px'}, children=[
            dcc.Graph(id='ecg-graph', animate=False)
        ]),
        
        # Right Panel: PCG
        html.Div(style={'width': '50%', 'padding': '10px'}, children=[
            dcc.Graph(id='pcg-graph', animate=False)
        ])
    ]),
    
    # Update both graphs every 100 milliseconds
    dcc.Interval(id='graph-update', interval=100, n_intervals=0)
])

@app.callback(
    [Output('ecg-graph', 'figure'),
     Output('pcg-graph', 'figure')],
    Input('graph-update', 'n_intervals')
)
def update_graphs(n):
    # --- ECG Figure Setup (Green) ---
    fig_ecg = go.Figure(data=go.Scattergl(
        y=list(ecg_buffer), mode='lines', line=dict(color='#00ff00', width=1.5)
    ))
    fig_ecg.update_layout(
        title='ECG Signal (Pin 35)',
        xaxis=dict(title='Samples', range=[0, BUFFER_SIZE], showgrid=False),
        yaxis=dict(title='Amplitude', autorange=True),
        template='plotly_dark',
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    # --- PCG Figure Setup (Cyan) ---
    fig_pcg = go.Figure(data=go.Scattergl(
        y=list(pcg_buffer), mode='lines', line=dict(color='#00ffff', width=1.5)
    ))
    fig_pcg.update_layout(
        title='PCG Audio (Pin 34)',
        xaxis=dict(title='Samples', range=[0, BUFFER_SIZE], showgrid=False),
        yaxis=dict(title='Amplitude', autorange=True),
        template='plotly_dark',
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    return fig_ecg, fig_pcg

if __name__ == '__main__':
    print("Starting Dual Dashboard. Open http://127.0.0.1:8050 in your browser.")
    app.run(debug=True, use_reloader=False)