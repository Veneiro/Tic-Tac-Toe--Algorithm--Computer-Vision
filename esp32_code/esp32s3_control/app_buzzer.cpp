#include "app_buzzer.h"

#define _COUNT(arr) (sizeof(arr) / sizeof(arr[0]))

// ── Constantes de notas (igual que aa.ino) ───────────────────────
namespace {
  const double C=261.63,Cs=277.18,D=293.66,Eb=311.13,E=329.63,F=349.23;
  const double Fs=369.99,G=392.00,Gs=415.30,A=440.00,Bb=466.16,B=493.88;
  const double S=0;

  // Octava 0
  const double C0=C/2,Cs0=Cs/2,D0=D/2,Eb0=Eb/2,E0=E/2,F0=F/2;
  const double Fs0=Fs/2,G0=G/2,Gs0=Gs/2,AA0=A/2,Bb0=Bb/2,BB0=B/2;

  // Octava 2
  const double C2=C*2,Cs2=Cs*2,D2=D*2,Eb2=Eb*2,E2=E*2,F2=F*2;
  const double Fs2=Fs*2,G2=G*2,Gs2=Gs*2,AA2=A*2,Bb2=Bb*2,B2=B*2;

  // Octava 3
  const double C3=C*4,Cs3=Cs*4,D3=D*4,Eb3=Eb*4,E3=E*4,F3=F*4;
  const double Fs3=Fs*4,G3=G*4,Gs3=Gs*4,AA3=A*4,Bb3=Bb*4,B3=B*4;
}

// ── Mario Kart DS - Waluigi Pinball  (BPM 135) ───────────────────
static const double _mk_notas1[] = {
  E,E2,A,B,D2,A,Bb,B,Cs2,B,
  E,E2,B,D2,E2,A,B,D2,Cs2,B,
  E,E2,B,D2,E2,A,Bb,B,Cs2,B,
  E,E2,E,D,D2,Cs,Cs2,BB0,B,S,
  E3,D3,B2,AA2,G2,E2,AA2,G2,AA2,B2,
  G2,E2,F2,Fs2,E2,S,E3,D3,S,B2,
  AA2,G2,E2,G2,E2,D2,Eb2,E2,S,E3,
  D3,B2,AA2,G2,F2,S,AA2,G2,AA2,B2,
  G2,E2,Fs2,E2,S,E3,Cs3,E3,Fs3,G3,
  E3,Cs3,E3,S,E3,D3,B2,AA2,F2,G2,
  E2,AA2,Cs3,E3,Fs3,G3,G3,G3,Fs3,E3,
  S,E3,D3,S,B2,AA2,G2,E2,G2,E2,
  Fs2,E2,Cs2,E2,S,E3,D3,B2,AA2,G2,
  E2,AA2,G2,AA2,B2,G2,E2,Fs2,E2,E3,
  Cs3,E3,G3,B3,AA3,G3,E3,D3,E3,D3,
  B2,D3,E3,E2,D2,B,D2,E2,B,D2,
  E2,Fs2,G2,Fs2,E2,E2,D2,B,D2,E2,
  D2,B,G,A,B,E,E3,D3,B2,D3,
  E3,B2,D3,E3,Fs3,G3,Fs3,E3,S,E3,
  Cs3,E3,G3,G3,B3,AA3,G3,E3,D3,E3,
  E,E2,A,B,D2,A,Bb,B,Cs2,B,
  E,E2,B,D2,E2,A,B,D2,Cs2,B,
  E,E2,B,D2,E2,A,Bb,B,Cs2,B,
  E,E2,E,D,D2,Cs,Cs2,BB0,B,E,
  E2,A,B,D2,A,Bb,B,Cs2,B,E,
  E2,B,D2,E2,A,B,D2,Cs2,B,E,
  E2,B,D2,E2,A,Bb,B,Cs2,B,E,
  E2,E,D,D2,Cs,Cs2,BB0,B,S,E3,
  D3,B2,AA2,G2,E2,AA2,G2,AA2,B2,G2,
  E2,F2,Fs2,E2,S,E3,D3,S,B2,AA2,
  G2,E2,G2,E2,D2,Eb2,E2,S,E3,D3,
  B2,AA2,G2,F2,S,AA2,G2,AA2,B2,G2,
  E2,Fs2,E2,S,E3,Cs3,E3,Fs3,G3,E3,
  Cs3,E3,S,E3,D3,B2,AA2,F2,G2,E2,
  AA2,Cs3,E3,Fs3,G3,G3,G3,Fs3,E3,S,
  E3,D3,S,B2,AA2,G2,E2,G2,E2,Fs2,
  E2,Cs2,E2,S,E3,D3,B2,AA2,G2,E2,
  AA2,G2,AA2,B2,G2,E2,Fs2,E2,E3,Cs3,
  E3,G3,B3,AA3,G3,E3,D3,E3,D3,B2,
  D3,E3,E2,D2,B,D2,E2,B,D2,E2,
  Fs2,G2,Fs2,E2,E2,D2,B,D2,E2,D2,
  B,G,A,B,E,E3,D3,B2,D3,E3,
  B2,D3,E3,Fs3,G3,Fs3,E3,S,E3,Cs3,
  E3,G3,G3,B3,AA3,G3,E3,D3,E3,S
};

static const double _mk_ritmo1[] = {
  0.5,0.5,0.5,0.25,0.5,0.25,0.25,0.25,0.5,0.5,
  0.5,0.5,0.5,0.25,0.5,0.25,0.25,0.25,0.5,0.5,
  0.5,0.5,0.5,0.25,0.5,0.25,0.25,0.25,0.5,0.5,
  0.5,0.25,0.25,0.5,0.5,0.5,0.5,0.5,0.4792,0.6458,
  0.4583,1.0,0.25,0.4167,0.75,0.4792,0.5208,0.5208,0.2708,0.4583,
  0.8125,0.4375,0.2604,0.2812,0.4167,0.5417,0.5,0.4792,0.5208,0.25,
  0.5,0.75,0.5,0.5,0.5,0.25,0.5,1.9792,0.7917,0.5,
  0.9792,0.2708,0.5,0.6042,0.0625,0.4792,0.5833,0.4583,0.2917,0.5,
  0.8125,0.5,0.4792,0.4167,0.5417,0.25,0.25,0.25,0.5,0.75,
  0.5,0.5,3.9792,1.125,0.5208,0.8958,0.3333,0.4792,0.1667,0.5,
  0.5208,0.5208,0.5208,0.5208,0.2708,0.4792,0.4375,0.2708,0.5,0.4375,
  0.5208,0.5,0.4583,0.5417,0.25,0.4792,0.7708,0.5,0.5,0.5,
  0.25,0.25,0.25,2.1667,0.6458,0.5417,0.8958,0.25,0.5,0.7917,
  0.5312,0.4896,0.4375,0.25,0.5,0.8333,0.5,0.5417,0.875,0.2292,
  0.2917,0.2708,0.4375,0.75,0.5,0.2292,0.3125,0.2604,0.6146,0.1562,
  0.1562,0.1667,3.1667,0.5208,0.5208,0.2708,0.4583,1.5,0.2708,0.5,
  0.8125,0.625,0.5417,0.7708,1.3542,0.4375,0.4583,0.25,0.3958,0.7708,
  0.5208,0.5,0.5,0.8125,0.6667,2.5,0.5,0.5,0.3333,0.4375,
  1.4792,0.25,0.5,0.8542,0.6667,0.5417,0.625,1.125,0.6875,0.25,
  0.25,0.25,0.625,0.125,0.5,0.5,0.25,0.25,0.25,4.25,
  0.5,0.5,0.5,0.25,0.5,0.25,0.25,0.25,0.5,0.5,
  0.5,0.5,0.5,0.25,0.5,0.25,0.25,0.25,0.5,0.5,
  0.5,0.5,0.5,0.25,0.5,0.25,0.25,0.25,0.5,0.5,
  0.5,0.25,0.25,0.5,0.5,0.5,0.5,0.5,0.5,0.5,
  0.5,0.5,0.25,0.5,0.25,0.25,0.25,0.5,0.5,0.5,
  0.5,0.5,0.25,0.5,0.25,0.25,0.25,0.5,0.5,0.5,
  0.5,0.5,0.25,0.5,0.25,0.25,0.25,0.5,0.5,0.5,
  0.25,0.25,0.5,0.5,0.5,0.5,0.5,0.4792,0.6458,0.4583,
  1.0,0.25,0.4167,0.75,0.4792,0.5208,0.5208,0.2708,0.4583,0.8125,
  0.4375,0.2604,0.2812,0.4167,0.5417,0.5,0.4792,0.5208,0.25,0.5,
  0.75,0.5,0.5,0.5,0.25,0.5,1.9792,0.7917,0.5,0.9792,
  0.2708,0.5,0.6042,0.0625,0.4792,0.5833,0.4583,0.2917,0.5,0.8125,
  0.5,0.4792,0.4167,0.5417,0.25,0.25,0.25,0.5,0.75,0.5,
  0.5,3.9792,1.125,0.5208,0.8958,0.3333,0.4792,0.1667,0.5,0.5208,
  0.5208,0.5208,0.5208,0.2708,0.4792,0.4375,0.2708,0.5,0.4375,0.5208,
  0.5,0.4583,0.5417,0.25,0.4792,0.7708,0.5,0.5,0.5,0.25,
  0.25,0.25,2.1667,0.6458,0.5417,0.8958,0.25,0.5,0.7917,0.5312,
  0.4896,0.4375,0.25,0.5,0.8333,0.5,0.5417,0.875,0.2292,0.2917,
  0.2708,0.4375,0.75,0.5,0.2292,0.3125,0.2604,0.6146,0.1562,0.1562,
  0.1667,3.1667,0.5208,0.5208,0.2708,0.4583,1.5,0.2708,0.5,0.8125,
  0.625,0.5417,0.7708,1.3542,0.4375,0.4583,0.25,0.3958,0.7708,0.5208,
  0.5,0.5,0.8125,0.6667,2.5,0.5,0.5,0.3333,0.4375,1.4792,
  0.25,0.5,0.8542,0.6667,0.5417,0.625,1.125,0.6875,0.25,0.25,
  0.25,0.625,0.125,0.5,0.5,0.25,0.25,0.25,4.0312,0.1775
};

static const double _mk_notas2[] = {
  E0,E,AA0,BB0,D,AA0,Bb0,BB0,Cs,BB0,
  E0,E,BB0,D,E,AA0,BB0,D,Cs,BB0,
  E0,E,BB0,D,E,AA0,Bb0,BB0,Cs,BB0,
  E,E2,E,D,D2,Cs,Cs2,BB0,B,E0,
  E,Gs0,D,E,AA0,Bb0,BB0,Cs,BB0,AA0,
  A,Cs,AA0,Cs,D,E,G,Fs,E,E0,
  E,Gs0,D,E,AA0,Bb0,BB0,Cs,BB0,AA0,
  A,AA0,Cs,Cs2,D,D2,Eb,Eb2,E0,E,
  Gs0,D,E,AA0,Bb0,BB0,Cs,BB0,AA0,A,
  Cs,AA0,Cs,D,E,G,Fs,E,AA0,A,
  Cs,G,A,BB0,Eb,Fs,B,BB0,E0,E,
  E0,Gs0,Gs,AA0,A,Bb0,Bb,E0,E,Gs0,
  D,E,AA0,Bb0,BB0,Cs,BB0,AA0,A,Cs,
  AA0,Cs,D,E,G,E,E0,E,Gs0,D,
  E,AA0,Bb0,BB0,Cs,BB0,AA0,A,AA0,Cs,
  Cs2,D,D2,Eb,Eb2,E0,E,Gs0,D,E,
  AA0,Bb0,BB0,Cs,BB0,AA0,A,Cs,AA0,Cs,
  D,E,G,Fs,E,AA0,A,Cs,G,A,
  D,Eb,E,Fs,E,E0,E,E0,Gs0,Gs,
  AA0,A,Bb0,Bb,G0,G,BB0,F,G,C,
  Cs,D,E,D,AA0,A,Cs,AA0,Cs,D,
  E,G,Fs,E,G0,G,BB0,F,G,C,
  Cs,D,E,D,Fs0,Fs,Fs0,Bb0,Fs,BB0,
  B,D,D2,G0,G,BB0,F,G,C,Cs,
  D,E,D,AA0,A,Cs,AA0,Cs,D,E,
  G,Fs,E,AA0,A,Cs,G,A,D,Eb,
  E,G,E,S,E,E2,E,D,D2,Cs,
  Cs2,BB0,B,E0,E,AA0,BB0,D,AA0,Bb0,
  BB0,Cs,BB0,E0,E,BB0,D,E,AA0,BB0,
  D,Cs,BB0,E0,E,BB0,D,E,AA0,Bb0,
  BB0,Cs,BB0,E,E2,E,D,D2,Cs,Cs2,
  BB0,B,E0,E,AA0,BB0,D,AA0,Bb0,BB0,
  Cs,BB0,E0,E,BB0,D,E,AA0,BB0,D,
  Cs,BB0,E0,E,BB0,D,E,AA0,Bb0,BB0,
  Cs,BB0,E,E2,E,D,D2,Cs,Cs2,BB0,
  B,E0,E,Gs0,D,E,AA0,Bb0,BB0,Cs,
  BB0,AA0,A,Cs,AA0,Cs,D,E,G,Fs,
  E,E0,E,Gs0,D,E,AA0,Bb0,BB0,Cs,
  BB0,AA0,A,AA0,Cs,Cs2,D,D2,Eb,Eb2,
  E0,E,Gs0,D,E,AA0,Bb0,BB0,Cs,BB0,
  AA0,A,Cs,AA0,Cs,D,E,G,Fs,E,
  AA0,A,Cs,G,A,BB0,Eb,Fs,B,BB0,
  E0,E,E0,Gs0,Gs,AA0,A,Bb0,Bb,E0,
  E,Gs0,D,E,AA0,Bb0,BB0,Cs,BB0,AA0,
  A,Cs,AA0,Cs,D,E,G,E,E0,E,
  Gs0,D,E,AA0,Bb0,BB0,Cs,BB0,AA0,A,
  AA0,Cs,Cs2,D,D2,Eb,Eb2,E0,E,Gs0,
  D,E,AA0,Bb0,BB0,Cs,BB0,AA0,A,Cs,
  AA0,Cs,D,E,G,Fs,E,AA0,A,Cs,
  G,A,D,Eb,E,Fs,E,E0,E,E0,
  Gs0,Gs,AA0,A,Bb0,Bb,G0,G,BB0,F,
  G,C,Cs,D,E,D,AA0,A,Cs,AA0,
  Cs,D,E,G,Fs,E,G0,G,BB0,F,
  G,C,Cs,D,E,D,Fs0,Fs,Fs0,Bb0,
  Fs,BB0,B,D,D2,G0,G,BB0,F,G,
  C,Cs,D,E,D,AA0,A,Cs,AA0,Cs,
  D,E,G,Fs,E,AA0,A,Cs,G,A,
  D,Eb,E,G,E,S,E,E2,E,D,
  D2,Cs,Cs2,BB0,B
};

static const double _mk_ritmo2[] = {
  0.5,0.5,0.5,0.25,0.5,0.25,0.25,0.25,0.5,0.5,
  0.5,0.5,0.5,0.25,0.5,0.25,0.25,0.25,0.5,0.5,
  0.5,0.5,0.5,0.25,0.5,0.25,0.25,0.25,0.5,0.5,
  0.5,0.25,0.25,0.5,0.5,0.5,0.5,0.5,0.5,0.5,
  0.5,0.5,0.25,0.5,0.25,0.25,0.25,0.5,0.5,0.5,
  0.5,0.5,0.25,0.5,0.25,0.25,0.25,0.5,0.5,0.5,
  0.5,0.5,0.25,0.5,0.25,0.25,0.25,0.5,0.5,0.5,
  0.25,0.25,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,
  0.5,0.25,0.5,0.25,0.25,0.25,0.5,0.5,0.5,0.5,
  0.5,0.25,0.5,0.25,0.25,0.25,0.5,0.5,0.5,0.5,
  0.5,0.25,0.5,0.25,0.25,0.25,0.5,0.5,0.5,0.25,
  0.25,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,
  0.25,0.5,0.25,0.25,0.25,0.5,0.5,0.5,0.5,0.5,
  0.25,0.5,0.25,0.25,0.75,0.5,0.5,0.5,0.5,0.25,
  0.5,0.25,0.25,0.25,0.5,0.5,0.5,0.25,0.25,0.5,
  0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.25,0.5,
  0.25,0.25,0.25,0.5,0.5,0.5,0.5,0.5,0.25,0.5,
  0.25,0.25,0.25,0.5,0.5,0.5,0.5,0.5,0.25,0.5,
  0.25,0.25,0.25,0.5,0.5,0.5,0.25,0.25,0.5,0.5,
  0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.25,0.5,0.25,
  0.25,0.25,0.5,0.5,0.5,0.5,0.5,0.25,0.5,0.25,
  0.25,0.25,0.5,0.5,0.5,0.5,0.5,0.25,0.5,0.25,
  0.25,0.25,0.5,0.5,0.5,0.25,0.25,0.5,0.5,0.5,
  0.5,0.5,0.5,0.5,0.5,0.5,0.25,0.5,0.25,0.25,
  0.25,0.5,0.5,0.5,0.5,0.5,0.25,0.5,0.25,0.25,
  0.25,0.5,0.5,0.5,0.5,0.5,0.25,0.5,0.25,0.25,
  0.25,0.5,0.35,0.15,0.5,0.25,0.25,0.5,0.5,0.5,
  0.5,0.5,0.5,0.5,0.5,0.5,0.25,0.5,0.25,0.25,
  0.25,0.5,0.5,0.5,0.5,0.5,0.25,0.5,0.25,0.25,
  0.25,0.5,0.5,0.5,0.5,0.5,0.25,0.5,0.25,0.25,
  0.25,0.5,0.5,0.5,0.25,0.25,0.5,0.5,0.5,0.5,
  0.5,0.5,0.5,0.5,0.5,0.25,0.5,0.25,0.25,0.25,
  0.5,0.5,0.5,0.5,0.5,0.25,0.5,0.25,0.25,0.25,
  0.5,0.5,0.5,0.5,0.5,0.25,0.5,0.25,0.25,0.25,
  0.5,0.5,0.5,0.25,0.25,0.5,0.5,0.5,0.5,0.5,
  0.5,0.5,0.5,0.5,0.25,0.5,0.25,0.25,0.25,0.5,
  0.5,0.5,0.5,0.5,0.25,0.5,0.25,0.25,0.25,0.5,
  0.5,0.5,0.5,0.5,0.25,0.5,0.25,0.25,0.25,0.5,
  0.5,0.5,0.25,0.25,0.5,0.5,0.5,0.5,0.5,0.5,
  0.5,0.5,0.5,0.25,0.5,0.25,0.25,0.25,0.5,0.5,
  0.5,0.5,0.5,0.25,0.5,0.25,0.25,0.25,0.5,0.5,
  0.5,0.5,0.5,0.25,0.5,0.25,0.25,0.25,0.5,0.5,
  0.5,0.25,0.25,0.5,0.5,0.5,0.5,0.5,0.5,0.5,
  0.5,0.5,0.25,0.5,0.25,0.25,0.75,0.5,0.5,0.5,
  0.5,0.25,0.5,0.25,0.25,0.25,0.5,0.5,0.5,0.25,
  0.25,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,
  0.25,0.5,0.25,0.25,0.25,0.5,0.5,0.5,0.5,0.5,
  0.25,0.5,0.25,0.25,0.25,0.5,0.5,0.5,0.5,0.5,
  0.25,0.5,0.25,0.25,0.25,0.5,0.5,0.5,0.25,0.25,
  0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.25,
  0.5,0.25,0.25,0.25,0.5,0.5,0.5,0.5,0.5,0.25,
  0.5,0.25,0.25,0.25,0.5,0.5,0.5,0.5,0.5,0.25,
  0.5,0.25,0.25,0.25,0.5,0.5,0.5,0.25,0.25,0.5,
  0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.25,0.5,
  0.25,0.25,0.25,0.5,0.5,0.5,0.5,0.5,0.25,0.5,
  0.25,0.25,0.25,0.5,0.5,0.5,0.5,0.5,0.25,0.5,
  0.25,0.25,0.25,0.5,0.35,0.15,0.5,0.25,0.25,0.5,
  0.5,0.5,0.5,0.5,0.4583
};

// ── New Super Mario Bros. Wii - Level Complete  (BPM 120) ────────
static const double _hw_notas1[] = {
  E2, S,  E2, E2, C2, E2, E2, G2, G2, C3,
  S,  C3
};
static const double _hw_ritmo1[] = {
  0.100, 0.150, 0.500, 0.500, 0.500, 0.500, 0.750, 0.500, 0.500, 0.100,
  0.150, 0.3729
};

static const double _hw_notas2[] = {
  G, A, Bb, B, C, S, C
};
static const double _hw_ritmo2[] = {
  0.750, 1.000, 1.250, 1.000, 0.100, 0.150, 0.3729
};

// ── New Super Mario Bros. Wii - Game Over  (BPM 145) ─────────────
static const double _go_notas1[] = {
  S,  F, G, G0, C
};
static const double _go_ritmo1[] = {
  0.0417, 2.375, 3.2083, 0.250, 10.125
};

static const double _go_notas2[] = {
  S,  AA2, C2, Eb2, D2, C2, A, B, C2
};
static const double _go_ritmo2[] = {
  0.0104, 1.375, 1.0208, 0.9688, 0.750, 0.750, 1.250, 0.125, 9.750
};

// ── Super Mario Bros. 3 - Boss Battle  (BPM 90) ──────────────────
static const double _bb_notas1[] = {
  S,  Fs, F,  E,  Eb, E0, E,  D2, E,  D2,
  Cs2,E,  C2, E,  C2, B,  C2, Cs2,E0, E,
  D2, E,  D2, Cs2,E,  C2, E,  C2, B,  C2,
  Cs2,E0, E,  D2, E,  D2, Cs2,E,  C2, E,
  C2, B,  C2, Cs2,E0, E,  D2, E,  D2, Cs2,
  E,  C2, E,  C2, B,  C2, Cs2,E0, E,  D2,
  E,  D2, Cs2,E,  C2, E,  C2, B,  C2, Cs2,
  F0, F,  Eb2,F,  Eb2,D2, F,  Cs2,F,  Cs2,
  C2, Cs2,D2, F0, F,  Eb2,F,  Eb2,D2, F,
  Cs2,F,  Cs2,C2, Cs2,D2, E0, E,  D2, E,
  D2, Cs2,E,  C2, E,  C2, B,  C2, Cs2,E0,
  E,  D2, E,  D2, Cs2,E,  C2, E,  C2, B,
  C2, Cs2,E0, E,  D2, E,  D2, Cs2,E,  C2,
  E,  C2, B,  C2, Cs2,E0, E,  D2, E,  D2,
  Cs2,E,  C2, E,  C2, B,  C2, Cs2,F0, F,
  Eb2,F,  Eb2,D2, F,  Cs2,F,  Cs2,C2, Cs2,
  D2, F0, F,  Eb2,F,  Eb2,D2, F,  Cs2,F,
  Cs2,C2, Cs2,D2
};
static const double _bb_ritmo1[] = {
  1.000,0.250,0.250,0.250,2.250,0.250,0.250,0.250,0.250,0.250,
  0.500,0.250,0.500,0.250,0.500,0.250,0.250,0.250,0.250,0.250,
  0.250,0.250,0.250,0.500,0.250,0.500,0.250,0.500,0.250,0.250,
  0.250,0.250,0.250,0.250,0.250,0.250,0.500,0.250,0.500,0.250,
  0.500,0.250,0.250,0.250,0.250,0.250,0.250,0.250,0.250,0.500,
  0.250,0.500,0.250,0.500,0.250,0.250,0.250,0.250,0.250,0.250,
  0.250,0.250,0.500,0.250,0.500,0.250,0.500,0.250,0.250,0.250,
  0.250,0.250,0.250,0.250,0.250,0.500,0.250,0.500,0.250,0.500,
  0.250,0.250,0.250,0.250,0.250,0.250,0.250,0.250,0.500,0.250,
  0.500,0.250,0.500,0.250,0.250,0.250,0.250,0.250,0.250,0.250,
  0.250,0.500,0.250,0.500,0.250,0.500,0.250,0.250,0.250,0.250,
  0.250,0.250,0.250,0.250,0.500,0.250,0.500,0.250,0.500,0.250,
  0.250,0.250,0.250,0.250,0.250,0.250,0.250,0.500,0.250,0.500,
  0.250,0.500,0.250,0.250,0.250,0.250,0.250,0.250,0.250,0.250,
  0.500,0.250,0.500,0.250,0.500,0.250,0.250,0.250,0.250,0.250,
  0.250,0.250,0.250,0.500,0.250,0.500,0.250,0.500,0.250,0.250,
  0.250,0.250,0.250,0.250,0.250,0.250,0.500,0.250,0.500,0.250,
  0.500,0.250,0.250,0.250
};

static const double _bb_notas2[] = {
  S,  C2, B,  Bb, A,  S,  B2, S,  AA2,G2,
  AA2,E2, G2, D2, B,  D2, E2, S,  B2, S,
  AA2,G2, AA2,E2, G2, D2, B,  D2, E2, S,
  C3, S,  B2, C3, F3, Eb3,B2, Bb2,Gs2,Bb2,
  Gs2,Bb2,F2, Gs2,F2, Eb2,F2, S,  B2, S,
  AA2,G2, AA2,E2, G2, D2, B,  D2, E2, S,
  B2, S,  AA2,G2, AA2,E2, G2, D2, B,  D2,
  E2, S,  C3, S,  B2, C3, F3, Eb3,B2, Bb2,
  Gs2,Bb2,Gs2,Bb2,F2, Gs2,F2, Eb2,F2, S
};
static const double _bb_ritmo2[] = {
  1.000,0.250,0.250,0.250,1.5729,5.1771,2.000,0.625,0.125,0.250,
  0.250,0.500,0.500,0.500,0.250,0.250,0.1979,2.5521,2.000,0.625,
  0.125,0.250,0.250,0.500,0.500,0.500,0.250,0.250,0.1979,2.5521,
  2.000,0.750,0.250,0.250,0.250,0.250,0.250,0.250,0.250,0.375,
  0.125,0.250,0.250,0.250,0.250,0.250,0.1979,1.5521,2.000,0.625,
  0.125,0.250,0.250,0.500,0.500,0.500,0.250,0.250,0.1979,2.5521,
  2.000,0.625,0.125,0.250,0.250,0.500,0.500,0.500,0.250,0.250,
  0.1979,2.5521,2.000,0.750,0.250,0.250,0.250,0.250,0.250,0.250,
  0.250,0.375,0.125,0.250,0.250,0.250,0.250,0.250,0.1979,1.0521
};

// ── Super Mario Kart - Race Fanfare  (BPM 122) ───────────────────
static const double _rf_notas1[] = {
  S, C, S, C, C, C, D, E, S
};
static const double _rf_ritmo1[] = {
  1.500000, 0.100000, 0.150000, 0.500000, 0.500000, 0.250000, 0.500000, 0.458300, 0.499900
};

static const double _rf_notas2[] = {
  A, C2, F2, AA2, F2, G2, S, G2, G2, G2, AA2, B2
};
static const double _rf_ritmo2[] = {
  0.250000, 0.250000, 0.333300, 0.333300, 0.333300, 0.100000, 0.150000, 0.500000, 0.500000, 0.250000, 0.500000, 0.958300
};

// ── Super Mario Bros. 3 - Game Over  (BPM 100) ───────────────────
static const double _dw_notas1[] = {
  A, S, D, D, D, Fs, G, S, Cs, D, G0
};
static const double _dw_ritmo1[] = {
  0.218800, 0.781200, 0.666700, 0.333300, 0.333300, 0.333300, 0.218800, 0.781200, 0.333300, 0.333300, 0.218800
};

static const double _dw_notas2[] = {
  E3, E3, E3, B3, AA3, Fs3, D3, G3, Cs2, D2, G, S
};
static const double _dw_ritmo2[] = {
  0.333300, 0.333300, 0.333300, 0.666700, 0.333300, 0.333300, 0.333300, 1.000000, 0.333300, 0.333300, 0.166700, 0.052200
};

// ── Tabla de canciones ────────────────────────────────────────────
struct CancionData {
    const double* notas1; int len1;
    const double* notas2; int len2;
    const double* ritmo1;
    const double* ritmo2;
    int bpm;
};

static const CancionData _tabla[NUM_CANCIONES] = {
    // CANCION_MARIO_KART
    {
        _mk_notas1, (int)_COUNT(_mk_notas1),
        _mk_notas2, (int)_COUNT(_mk_notas2),
        _mk_ritmo1, _mk_ritmo2,
        135
    },
    // CANCION_HUMAN_WIN
    {
        _hw_notas1, (int)_COUNT(_hw_notas1),
        _hw_notas2, (int)_COUNT(_hw_notas2),
        _hw_ritmo1, _hw_ritmo2,
        120
    },
    // CANCION_ROBOT_WIN
    {
        _go_notas1, (int)_COUNT(_go_notas1),
        _go_notas2, (int)_COUNT(_go_notas2),
        _go_ritmo1, _go_ritmo2,
        145
    },
    // CANCION_BOSS_BATTLE
    {
        _bb_notas1, (int)_COUNT(_bb_notas1),
        _bb_notas2, (int)_COUNT(_bb_notas2),
        _bb_ritmo1, _bb_ritmo2,
        90
    },
    // CANCION_RACE_FANFARE
    {
        _rf_notas1, (int)_COUNT(_rf_notas1),
        _rf_notas2, (int)_COUNT(_rf_notas2),
        _rf_ritmo1, _rf_ritmo2,
        122
    },
    // CANCION_DRAW
    {
        _dw_notas1, (int)_COUNT(_dw_notas1),
        _dw_notas2, (int)_COUNT(_dw_notas2),
        _dw_ritmo1, _dw_ritmo2,
        100
    }
};

// ── Estado interno ────────────────────────────────────────────────
enum EstadoBuzzer { BZ_STOP, BZ_PLAY, BZ_PAUSE };

struct Voz {
    int pin;
    const double* notas;
    const double* ritmo;
    int totalNotas;
    int notaActual;
    unsigned long tiempoInicioNota;
    int msPorNegra;
};

static Voz            _v1, _v2;
static volatile EstadoBuzzer _estado   = BZ_STOP;
static volatile bool         _repetir  = true;

/** @brief Silencia una voz escribiendo frecuencia 0 en su canal LEDC.
 *  @param v Referencia a la Voz que se desea silenciar. */
static void _silenciar(Voz &v) { ledcWriteTone(v.pin, 0); }

/** @brief Avanza la reproducción de notas para una voz; debe llamarse cada 1 ms.
 *  Aplica un silencio de corte del 10% al final antes de pasar a la siguiente nota.
 *  Gestiona el bucle y la detección de fin de canción.
 *  @param v Referencia a la Voz cuyo estado de reproducción se actualiza. */
static void _tocarVoz(Voz &v) {
    unsigned long ahora   = millis();
    unsigned long elapsed = ahora - v.tiempoInicioNota;
    int duracion = (int)(v.ritmo[v.notaActual] * v.msPorNegra);
    int corte    = duracion * 9 / 10;

    if (elapsed >= (unsigned long)corte && elapsed < (unsigned long)duracion) {
        ledcWriteTone(v.pin, 0);
    }

    if (elapsed >= (unsigned long)duracion) {
        v.notaActual++;
        if (v.notaActual >= v.totalNotas) {
            if (_repetir) {
                v.notaActual = 0;
            } else {
                v.notaActual = v.totalNotas;
                _silenciar(v);
                return;
            }
        }
        double freq = v.notas[v.notaActual];
        ledcWriteTone(v.pin, freq > 0 ? (uint32_t)freq : 0);
        v.tiempoInicioNota = ahora;
    }
}

/** @brief Devuelve verdadero cuando ambas voces han terminado todas las notas en modo no repetición.
 *  @return Verdadero si _v1 y _v2 han llegado al final de sus arrays de notas. */
static bool _ambasTerminaron() {
    return !_repetir &&
           _v1.notaActual >= _v1.totalNotas &&
           _v2.notaActual >= _v2.totalNotas;
}

// ── Tarea FreeRTOS (núcleo 0) ─────────────────────────────────────
/** @brief Tarea FreeRTOS fijada al núcleo 0 que gestiona ambas voces del buzzer.
 *  Llama a _tocarVoz para cada voz cada 1 ms y detiene la reproducción cuando ambas terminan.
 *  @param params Parámetro de tarea no utilizado (pasar nullptr). */
static void _buzzerTask(void* /*params*/) {
    while (true) {
        if (_estado == BZ_PLAY) {
            _tocarVoz(_v1);
            _tocarVoz(_v2);
            if (_ambasTerminaron()) _estado = BZ_STOP;
        }
        vTaskDelay(pdMS_TO_TICKS(1));
    }
}

// ── API pública ───────────────────────────────────────────────────
/** @brief Inicializa los canales LEDC para ambos buzzers y lanza la tarea FreeRTOS de reproducción.
 *  El canal 0 (10 bits) controla BUZZER_PIN_1; el canal 1 (8 bits) controla BUZZER_PIN_2.
 *  Usa canales y resoluciones separados para evitar que compartan temporizador con el servo. */
void buzzerInit() {
    // Canales y resoluciones distintos fuerzan timers LEDC separados,
    // evitando que ledcWriteTone de una voz cambie la frecuencia de la otra
    // y que el servo (canal auto-asignado ≥2) comparta timer con los buzzers.
    ledcAttachChannel(BUZZER_PIN_1, 1000, 10, 0);
    ledcAttachChannel(BUZZER_PIN_2, 1000,  8, 1);
    ledcWriteTone(BUZZER_PIN_1, 0);
    ledcWriteTone(BUZZER_PIN_2, 0);
    xTaskCreatePinnedToCore(_buzzerTask, "buzzer", 3072, nullptr, 1, nullptr, 0);
}

/** @brief Comienza a reproducir una canción de la tabla de canciones, deteniendo primero cualquier reproducción en curso.
 *  @param id    Identificador de la canción a reproducir (índice en _tabla[]).
 *  @param repetir Verdadero para repetir la canción indefinidamente; falso para reproducirla una sola vez. */
void buzzerPlay(CancionId id, bool repetir) {
    if (id < 0 || id >= NUM_CANCIONES) return;
    const CancionData &c = _tabla[id];

    _estado = BZ_STOP;
    _silenciar(_v1);
    _silenciar(_v2);
    vTaskDelay(pdMS_TO_TICKS(2));

    _v1 = { BUZZER_PIN_1, c.notas1, c.ritmo1, c.len1, 0, millis(), (int)(60000.0 / c.bpm) };
    _v2 = { BUZZER_PIN_2, c.notas2, c.ritmo2, c.len2, 0, millis(), (int)(60000.0 / c.bpm) };

    ledcWriteTone(BUZZER_PIN_1, _v1.notas[0] > 0 ? (uint32_t)_v1.notas[0] : 0);
    ledcWriteTone(BUZZER_PIN_2, _v2.notas[0] > 0 ? (uint32_t)_v2.notas[0] : 0);

    _repetir = repetir;
    _estado  = BZ_PLAY;
}

/** @brief Detiene la reproducción inmediatamente y rebobina ambas voces a la nota 0. */
void buzzerStop() {
    _estado = BZ_STOP;
    _silenciar(_v1);
    _silenciar(_v2);
    _v1.notaActual = 0;
    _v2.notaActual = 0;
}

/** @brief Pausa la reproducción si el buzzer está sonando actualmente; no tiene efecto en caso contrario. */
void buzzerPause() {
    if (_estado != BZ_PLAY) return;
    _estado = BZ_PAUSE;
    _silenciar(_v1);
    _silenciar(_v2);
}

/** @brief Reanuda la reproducción desde un estado pausado; reinicia el tiempo de inicio de nota al momento actual.
 *  No tiene efecto si el buzzer no está pausado. */
void buzzerResume() {
    if (_estado != BZ_PAUSE) return;
    unsigned long ahora = millis();
    _v1.tiempoInicioNota = ahora;
    _v2.tiempoInicioNota = ahora;
    ledcWriteTone(BUZZER_PIN_1, _v1.notas[_v1.notaActual] > 0 ? (uint32_t)_v1.notas[_v1.notaActual] : 0);
    ledcWriteTone(BUZZER_PIN_2, _v2.notas[_v2.notaActual] > 0 ? (uint32_t)_v2.notas[_v2.notaActual] : 0);
    _estado = BZ_PLAY;
}

/** @brief Devuelve verdadero si el buzzer se encuentra actualmente en estado BZ_PLAY.
 *  @return Verdadero cuando está reproduciendo activamente, falso cuando está detenido o pausado. */
bool buzzerSonando() {
    return _estado == BZ_PLAY;
}

/** @brief Devuelve la duración total en milisegundos de la voz más larga de una canción.
 *  @param id Identificador de la canción a consultar.
 *  @return Duración en ms de la más larga de las dos voces; 0 si el identificador es inválido. */
unsigned long buzzerDuracion(CancionId id) {
    if (id < 0 || id >= NUM_CANCIONES) return 0;
    const CancionData &c = _tabla[id];
    int msPorNegra = (int)(60000.0 / c.bpm);
    double dur1 = 0, dur2 = 0;
    for (int i = 0; i < c.len1; i++) dur1 += c.ritmo1[i];
    for (int i = 0; i < c.len2; i++) dur2 += c.ritmo2[i];
    double durMax = dur1 > dur2 ? dur1 : dur2;
    return (unsigned long)(durMax * msPorNegra);
}
