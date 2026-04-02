/*
  Timed Laboratory Exercise #1 - Task 2
  8-Key Tone Generator with 7-Segment Display
  Name: De Silva B. K. P.
  Index Number: 230123K
*/

// 7-segment display pins (segments a, b, c, d, e, f, g)
const int segPins[7] = {2, 3, 4, 5, 6, 7, 8}; // a=D2, b=D3 ... g=D8

// Button pins (using analog pins as digital)
const int buttonPins[8] = {A0, A1, A2, A3, A4, A5, 11, 12};

// Speaker pin
const int speakerPin = 10;

// Frequencies
const int frequencies[8] = {300, 400, 500, 600, 700, 800, 900, 1000};

// 7-segment encoding for digits 1–8 (common cathode: HIGH = segment ON)
// Segments: a, b, c, d, e, f, g
const byte digitMap[9][7] = {
  //a  b  c  d  e  f  g
  {0, 0, 0, 0, 0, 0, 0}, // 0 (unused situation)
  {0, 1, 1, 0, 0, 0, 0}, // 1
  {1, 1, 0, 1, 1, 0, 1}, // 2
  {1, 1, 1, 1, 0, 0, 1}, // 3
  {0, 1, 1, 0, 0, 1, 1}, // 4
  {1, 0, 1, 1, 0, 1, 1}, // 5
  {1, 0, 1, 1, 1, 1, 1}, // 6
  {1, 1, 1, 0, 0, 0, 0}, // 7
  {1, 1, 1, 1, 1, 1, 1}, // 8
};

void displayDigit(int num) {
  if (num < 0 || num > 8) return;
  for (int i = 0; i < 7; i++) {
    digitalWrite(segPins[i], digitMap[num][i]);
  }
}

void clearDisplay() {
  for (int i = 0; i < 7; i++) {
    digitalWrite(segPins[i], LOW);
  }
}

void setup() {
  for (int i = 0; i < 7; i++) {
    pinMode(segPins[i], OUTPUT);
  }
  for (int i = 0; i < 8; i++) {
    pinMode(buttonPins[i], INPUT); // external pull-down resistors
  }
  pinMode(speakerPin, OUTPUT);
  clearDisplay();
}

void loop() {
  int totalFreq = 0;
  int pressedKeys[8];
  int pressedCount = 0;

  // Detect pressed keys
  for (int i = 0; i < 8; i++) {
    if (digitalRead(buttonPins[i]) == HIGH) {
      totalFreq += frequencies[i];
      pressedKeys[pressedCount++] = i + 1; // Key numbers 1–8
    }
  }

  // Handle tone
  if (pressedCount > 0) {
    tone(speakerPin, totalFreq);
  } else {
    noTone(speakerPin);
    clearDisplay();
    return;
  }

  // Display logic
  if (pressedCount == 1) {
    displayDigit(pressedKeys[0]);
    delay(20); // Minimum tone duration
  } else {
    // Alternate between pressed key numbers at 500ms intervals
    static int altIndex = 0;
    static unsigned long lastSwitch = 0;

    if (millis() - lastSwitch >= 500) {
      altIndex = (altIndex + 1) % pressedCount;
      lastSwitch = millis();
    }
    displayDigit(pressedKeys[altIndex]);
    delay(20);
  }
}