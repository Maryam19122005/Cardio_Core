// Define the ADC1 pins
const int ECG_PIN = 35; 
const int PCG_PIN = 34; 

// ECG Leads-off detection pins
const int LO_PLUS_PIN  = 4;  
const int LO_MINUS_PIN = 5;  

void setup() {
  Serial.begin(115200); 

  pinMode(LO_PLUS_PIN, INPUT);
  pinMode(LO_MINUS_PIN, INPUT);
  
  // Set the internal ADC to 12-bit resolution
  analogReadResolution(12); 
}

void loop() {
  int ecgValue = 0;

  // 1. Read ECG (with flatline fallback if leads are disconnected)
  if ((digitalRead(LO_PLUS_PIN) == 1) || (digitalRead(LO_MINUS_PIN) == 1)) {
    ecgValue = 0; 
  } else {
    ecgValue = analogRead(ECG_PIN);
  }

  // 2. Read PCG 
  int pcgValue = analogRead(PCG_PIN);

  // 3. Print as comma-separated values: "ECG,PCG"
  Serial.print(ecgValue);
  Serial.print(",");
  Serial.println(pcgValue);

  // ~1ms delay for a 1000 Hz sampling rate
  delay(1);
}