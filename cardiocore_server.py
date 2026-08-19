"""
CardioCore Flask Server
Provides /data endpoint for Unity VR app
Handles mock data, real ESP32 streams, and demo modes
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import threading
import time
import json
from datetime import datetime

from cardiocore_constants import Rhythm, Murmur
from mock_data_generator import MockDataGenerator
from fusion_engine import CardioFusionEngine
from signal_buffers import FrameParser

# ============ INITIALIZATION ============
app = Flask(__name__)
CORS(app)

# Global state
engine = CardioFusionEngine()
mock_gen = MockDataGenerator(bpm=72, rhythm=Rhythm.NORMAL, murmur=Murmur.NONE)
data_lock = threading.Lock()

# Server state
server_state = {
    'mode': 'mock',  # 'mock' or 'live'
    'running': True,
    'last_state': None,
    'last_update_ms': 0,
    'frame_count': 0,
    'error_count': 0,
    'demo_rhythm': Rhythm.NORMAL,
    'demo_murmur': Murmur.NONE,
}

# ============ BACKGROUND DATA GENERATION ============
def data_generation_thread():
    """
    Background thread continuously generates/processes frames
    Updates shared state
    """
    frame_count = 0
    
    while server_state['running']:
        try:
            # Generate frame based on mode
            if server_state['mode'] == 'mock':
                # Update mock generator if demo mode changed
                mock_gen.rhythm = server_state['demo_rhythm']
                mock_gen.murmur = server_state['demo_murmur']
                frame = mock_gen.generate_frame()
            else:
                # In live mode, would read from serial here
                # For now, continue using mock
                frame = mock_gen.generate_frame()
            
            # Process frame through fusion engine
            twin_state = engine.process_frame(frame)
            
            if twin_state:
                with data_lock:
                    server_state['last_state'] = twin_state
                    server_state['last_update_ms'] = int(time.time() * 1000)
                    server_state['frame_count'] += 1
            
            # Sleep to maintain 10ms frame interval (100 frames/sec)
            time.sleep(0.01)
            
        except Exception as e:
            with data_lock:
                server_state['error_count'] += 1
            print(f"ERROR in data generation: {e}")
            time.sleep(0.01)

# Start background thread
data_thread = threading.Thread(target=data_generation_thread, daemon=True)
data_thread.start()
print("✓ Background data thread started")

# ============ ROUTES ============

@app.route('/data', methods=['GET'])
def get_data():
    """
    Main endpoint: returns current cardiac state
    Called by Unity app (likely 30-60 Hz)
    
    Response format (JSON):
    {
        'timestamp_ms': int,
        'ecg': float,
        'pcg': float,
        'bpm': int,
        'cardiac_phase': float (0.0-1.0),
        'systole_phase': float,
        'diastole_phase': float,
        'rhythm': str,
        'murmur': str,
        'lead_off': bool,
        ...
    }
    """
    with data_lock:
        if server_state['last_state'] is None:
            return jsonify({
                'error': 'No data available yet',
                'uptime_ms': int(time.time() * 1000)
            }), 202  # Accepted but not ready
        
        state = server_state['last_state']
        return jsonify(state.to_dict()), 200

@app.route('/status', methods=['GET'])
def get_status():
    """
    Server health check
    """
    with data_lock:
        uptime = int(time.time() * 1000) - server_state['last_update_ms']
        return jsonify({
            'mode': server_state['mode'],
            'running': server_state['running'],
            'frames_processed': server_state['frame_count'],
            'errors': server_state['error_count'],
            'last_update_ms_ago': uptime,
            'demo_rhythm': server_state['demo_rhythm'].value,
            'demo_murmur': server_state['demo_murmur'].value,
        }), 200

@app.route('/command', methods=['POST'])
def command():
    """
    Control server behavior
    
    POST body:
    {
        'action': str,  # 'set_rhythm', 'set_murmur', 'switch_mode'
        'value': str    # 'normal', 'brady', 'tachy', 'afib', etc.
    }
    """
    data = request.get_json()
    action = data.get('action', '')
    value = data.get('value', '')
    
    try:
        with data_lock:
            if action == 'set_rhythm':
                for rhythm in Rhythm:
                    if rhythm.value == value:
                        server_state['demo_rhythm'] = rhythm
                        return jsonify({
                            'success': True,
                            'message': f'Rhythm set to {value}'
                        }), 200
                return jsonify({
                    'success': False,
                    'message': f'Unknown rhythm: {value}'
                }), 400
            
            elif action == 'set_murmur':
                for murmur in Murmur:
                    if murmur.value == value:
                        server_state['demo_murmur'] = murmur
                        return jsonify({
                            'success': True,
                            'message': f'Murmur set to {value}'
                        }), 200
                return jsonify({
                    'success': False,
                    'message': f'Unknown murmur: {value}'
                }), 400
            
            elif action == 'switch_mode':
                if value in ['mock', 'live']:
                    server_state['mode'] = value
                    return jsonify({
                        'success': True,
                        'message': f'Mode switched to {value}'
                    }), 200
                return jsonify({
                    'success': False,
                    'message': f'Unknown mode: {value}'
                }), 400
            
            else:
                return jsonify({
                    'success': False,
                    'message': f'Unknown action: {action}'
                }), 400
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/upload_frame', methods=['POST'])
def upload_frame():
    """
    Accept raw binary frames from ESP32 (future use)
    POST body: raw binary (46 bytes per frame)
    """
    try:
        binary_data = request.data
        
        if len(binary_data) != 46:
            return jsonify({
                'success': False,
                'message': f'Expected 46 bytes, got {len(binary_data)}'
            }), 400
        
        frame = FrameParser.unpack_frame(binary_data)
        if frame is None:
            return jsonify({
                'success': False,
                'message': 'Failed to parse frame'
            }), 400
        
        valid, msg = FrameParser.validate_frame(frame)
        if not valid:
            return jsonify({
                'success': False,
                'message': f'Frame validation failed: {msg}'
            }), 400
        
        # Process frame
        twin_state = engine.process_frame(frame)
        
        with data_lock:
            server_state['last_state'] = twin_state
            server_state['frame_count'] += 1
        
        return jsonify({
            'success': True,
            'frame_number': frame.frame_number
        }), 200
    
    except Exception as e:
        with data_lock:
            server_state['error_count'] += 1
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/debug', methods=['GET'])
def debug():
    """Debug endpoint: returns detailed state (verbose)"""
    with data_lock:
        if server_state['last_state'] is None:
            return jsonify({'error': 'No data'}), 202
        
        state = server_state['last_state']
        return jsonify({
            'timestamp_ms': state.timestamp_ms,
            'ecg': float(state.ecg),
            'pcg': float(state.pcg),
            'bpm': state.bpm,
            'last_r_peak_ms': state.last_r_peak_ms,
            'last_s1_ms': state.last_s1_ms,
            'last_s2_ms': state.last_s2_ms,
            'cardiac_phase': round(state.cardiac_phase, 4),
            'systole_phase': round(state.systole_phase, 4),
            'diastole_phase': round(state.diastole_phase, 4),
            'rhythm': state.rhythm.value,
            'murmur': state.murmur.value,
            'lead_off': state.lead_off,
            'ecg_confidence': round(state.ecg_confidence, 3),
            'pcg_confidence': round(state.pcg_confidence, 3),
            'server_frames_processed': server_state['frame_count'],
            'server_mode': server_state['mode'],
        }), 200

# ============ APP STARTUP/SHUTDOWN ============

@app.before_request
def before_request():
    """Log requests (debug)"""
    pass

@app.teardown_appcontext
def shutdown(exception=None):
    """Cleanup on shutdown"""
    server_state['running'] = False

@app.after_request
def close_connection(response):
    response.headers['Connection'] = 'close'
    return response
# ============ RUN SERVER ============

if __name__ == '__main__':
    print("\n" + "="*60)
    print("CardioCore Flask Server")
    print("="*60)
    print("\nEndpoints:")
    print("  GET  /data              - Latest cardiac state (for Unity)")
    print("  GET  /status            - Server health")
    print("  POST /command           - Control (set_rhythm, set_murmur, etc)")
    print("  POST /upload_frame      - Upload binary frame (ESP32)")
    print("  GET  /debug             - Verbose state (development)")
    print("\nStarting server on http://0.0.0.0:5000")
    print("="*60 + "\n")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,  # Set to True only for development
        use_reloader=False  # Prevent threading issues
    )
