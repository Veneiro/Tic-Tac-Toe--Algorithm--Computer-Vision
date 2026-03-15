//****************LIBRERIAS*****************

#include <Wire.h>

#include <LiquidCrystal_I2C.h>

//******************************************



//Definicion de variables

#define LCD_ADDRESS 0x27  //direccion de memoria del LCD

#define LCD_COLUMNS 20    //numero de columnas del LCD

#define LCD_ROWS 4        //numero de filas del LCD

//Defino los objetos que voy a utilizar

LiquidCrystal_I2C lcd(LCD_ADDRESS, LCD_COLUMNS, LCD_ROWS);    //creamos un objeto llamado lcd



int inPin = 5;

const int pinSwitch=10;

const int pinStart=8;

const int pinMenu=42;







// Definicion de variables globales

int modo = 0; // 0 = Automático, 1 = Manual

int ganador=0; // Se inicia sin ganador

int tablero[3][3]={{0,0,0},{0,0,0},{0,0,0}};

///////////



String mensaje= "";





bool modoAutomatico = false;

bool juegoEnCurso = false;





void setup() {



  //**********INICIALIZACION*****************

  Serial.begin(115200);

  pinMode(inPin, INPUT_PULLUP);

  Wire.begin(7,9,10000);



  lcd.begin(LCD_COLUMNS,LCD_ROWS,LCD_ADDRESS);

  lcd.init();

  lcd.backlight();

  lcd.clear();







  pinMode(pinSwitch, INPUT_PULLUP);

  pinMode(pinStart, INPUT_PULLUP);

  pinMode(pinMenu, INPUT_PULLUP);

//*********************************************

 mostrarBienvenida();

 





}



void loop() {

  // 1. Fase de menu

  esperarSeleccionMenu();



  //  2. Fase de juego

  ejecutarPartida1();





 

}



void mostrarBienvenida() {

  lcd.clear();

  lcd.setCursor(2, 0);

  lcd.print("ROBOT TIC-TAC-TOE");

  lcd.setCursor(0, 1);

  lcd.print("--------------------");

  delay(3000);

}



void esperarSeleccionMenu() {

  bool ultimoEstadoSwitch = !digitalRead(pinSwitch); // Forzar primera actualización

 

  lcd.clear();

  lcd.setCursor(5, 0); lcd.print("MENU ROBOT");

  lcd.setCursor(0, 1); lcd.print("--------------------");



  // Bucle infinito hasta que se pulse START

  while (digitalRead(pinStart) == LOW) {

    bool lecturaSwitch = digitalRead(pinSwitch);



    // Si el switch cambia de posición, actualizamos el texto

    if (lecturaSwitch != ultimoEstadoSwitch) {

      lcd.setCursor(0, 2);

      if (lecturaSwitch == LOW) { // Switch en ON

        lcd.print(" > MODO: AUTOMATICO ");

        modoAutomatico = true;

      } else {

        lcd.print(" > MODO: MANUAL     ");

        modoAutomatico = false;

      }

      ultimoEstadoSwitch = lecturaSwitch;

      delay(50); // Debounce

    }



    // Efecto visual: "Pulse Start" parpadeando sin detener el programa

    if ((millis() / 600) % 2 == 0) {

      lcd.setCursor(2, 3);

      lcd.print(">> PULSE START <<");

    } else {

      lcd.setCursor(2, 3);

      lcd.print("                 ");

    }

  }



  // Al salir del bucle (se pulsó Start)

  confirmarInicio();

}



void confirmarInicio() {

  lcd.clear();

  lcd.setCursor(3, 1);

  lcd.print("CONFIGURADO OK");

  lcd.setCursor(0, 2);

  lcd.print(modoAutomatico ? "INICIANDO AUTOMATICO" : "INICIANDO MANUAL");

 


  delay(1000);

}


void ejecutarPartida(){
vaciarTablero();
lcd.clear(); // Limpio lcd
lcd.setCursor(1,0);//Posiciono en el primer pixel
lcd.print("ESTADO DEL TABLERO");
int ganador = comprobarGanador();
  if(modoAutomatico == true){
      actualizarLCD();
    while(ganador==0){
      // SE SIGUE EL JUEGO
      
      
      if (Serial.available() > 0) {
    
      // Leemos el texto hasta encontrar un salto de línea (Enter)
       mensaje = Serial.readStringUntil('\n');
    
      // Limpiamos espacios en blanco o caracteres raros al final
      mensaje.trim(); 
      leetablero(mensaje);
      actualizarLCD();
      ganador = comprobarGanador();
    }

  }
  if(ganador==1){
    //GANA EL JUGADOR 1
    delay(1500);
    lcd.clear();
    lcd.setCursor(0,0);//Posiciono en el primer pixel
    lcd.print("********************");
    lcd.setCursor(6,1);//Posiciono en el primer pixel
    lcd.print("GANADOR");
    lcd.setCursor(5,2);//Posiciono en el primer pixel
    lcd.print("JUGADOR 1");
    lcd.setCursor(0,3);//Posiciono en el primer pixel
    lcd.print("********************");
    delay(3000);
  }else if(ganador==2){
    //GANA EL JUGADOR 2
      delay(1500);
    lcd.clear();
    lcd.setCursor(0,0);//Posiciono en el primer pixel
    lcd.print("********************");
    lcd.setCursor(6,1);//Posiciono en el primer pixel
    lcd.print("GANADOR");
    lcd.setCursor(5,2);//Posiciono en el primer pixel
    lcd.print("JUGADOR 2");
    lcd.setCursor(0,3);//Posiciono en el primer pixel
    lcd.print("********************");
    delay(3000);
  }else if(ganador==3){
    //EMPATE
    delay(1500);
    lcd.clear();
    lcd.setCursor(0,0);//Posiciono en el primer pixel
    lcd.print("********************");
    lcd.setCursor(3,1);//Posiciono en el primer pixel
    lcd.print("NO HAY GANADOR");
    lcd.setCursor(7,2);//Posiciono en el primer pixel
    lcd.print("EMPATE");
    lcd.setCursor(0,3);//Posiciono en el primer pixel
    lcd.print("********************");
    delay(3000);
  }
    
  
    lcd.clear();
    lcd.setCursor(0,0);//Posiciono en el primer pixel
    lcd.print("********************");
    lcd.setCursor(3,1);//Posiciono en el primer pixel
    lcd.print("PARTIDA ACABADA");
    lcd.setCursor(0,2);//Posiciono en el primer pixel
    lcd.print("VOLVIENDO AL MENU...");
    lcd.setCursor(0,3);//Posiciono en el primer pixel
    lcd.print("********************");
    delay(3000);
    lcd.clear();


  }else if(modoAutomatico == false){
    //Se ejecuta el modo manual


  }




}
void ejecutarPartida1() {
  vaciarTablero();
  lcd.clear(); 
  lcd.setCursor(1,0);
  lcd.print("ESTADO DEL TABLERO");
  
  int ganador = comprobarGanador();
  bool abortarPartida = false; // Flag para saber si queremos salir del juego
  static bool ultimoEstadoMenu = HIGH; // Para detectar el flanco del botón pausa

  if (modoAutomatico == true) {
    actualizarLCD();
    
    // El juego sigue mientras no haya ganador Y no hayamos pulsado salir en la pausa
    while (ganador == 0 && !abortarPartida) {
      
      // --- 1. DETECCIÓN DE FLANCO PARA PAUSA ---
      bool lecturaMenu = digitalRead(pinMenu);
      
      if (lecturaMenu == LOW && ultimoEstadoMenu == HIGH) { // Flanco de bajada
        delay(50); // Debounce
        
        // Entramos a la función de pausa y guardamos si el usuario decidió salir
        abortarPartida = abrirMenuPausa(); 
        
        if (!abortarPartida) {
          // Si regresamos a la partida, redibujamos el estado del tablero
          lcd.clear();
          lcd.setCursor(1,0);
          lcd.print("ESTADO DEL TABLERO");
          actualizarLCD();
        }
      }
      ultimoEstadoMenu = lecturaMenu;

      // --- 2. LÓGICA DE JUEGO (SERIAL) ---
      if (Serial.available() > 0 && !abortarPartida) {
        mensaje = Serial.readStringUntil('\n');
        mensaje.trim(); 
        leetablero(mensaje);
        actualizarLCD();
        ganador = comprobarGanador();
      }
    }

    // --- 3. FINALIZACIÓN ---
    if (abortarPartida) {
       return; // Sale directamente de ejecutarPartida y vuelve al loop (Menu)
    }

    // Si salimos por ganador (no por abortar), mostramos quién ganó
    manejarFinDeJuego(ganador);

  } else {
    // Modo manual...
  }
}

// NUEVA FUNCIÓN: Maneja el estado de pausa
bool abrirMenuPausa() {
  
  lcd.clear();
  lcd.setCursor(7, 1); lcd.print("PAUSA");
  lcd.setCursor(0, 2); lcd.print("START: Salir");
  lcd.setCursor(0, 3); lcd.print("MENU: Continuar");

  static bool antStartPausa = LOW;
  static bool antMenuPausa = LOW;

  while (true) {
    bool actStart = digitalRead(pinStart);
    bool actMenu = digitalRead(pinMenu);

    // FLANCO EN START -> SALIR DE LA PARTIDA
    if (actStart == HIGH && antStartPausa == LOW) {
      delay(50);
      return true; // Retornamos true para indicar que queremos ABORTAR
    }
    antStartPausa = actStart;

    // FLANCO EN MENU -> CONTINUAR PARTIDA
    if (actMenu == HIGH && antMenuPausa == LOW) {
      delay(50);
      return false; // Retornamos false para indicar que NO queremos abortar (continuar)
    }
    antMenuPausa = actMenu;
  }
}

// Función auxiliar para no repetir los mensajes de ganador
void manejarFinDeJuego(int ganador) {
  if(ganador >= 1 && ganador <= 3) {
    delay(1500);
    lcd.clear();
    lcd.setCursor(0,0); lcd.print("********************");
    
    if (ganador == 1) {
      lcd.setCursor(6,1); lcd.print("GANADOR");
      lcd.setCursor(5,2); lcd.print("JUGADOR 1");
    } else if (ganador == 2) {
      lcd.setCursor(6,1); lcd.print("GANADOR");
      lcd.setCursor(5,2); lcd.print("JUGADOR 2");
    } else {
      lcd.setCursor(3,1); lcd.print("NO HAY GANADOR");
      lcd.setCursor(7,2); lcd.print("EMPATE");
    }
    
    lcd.setCursor(0,3); lcd.print("********************");
    delay(3000);
    
    // Mensaje final
    lcd.clear();
    lcd.setCursor(0,0); lcd.print("********************");
    lcd.setCursor(3,1); lcd.print("PARTIDA ACABADA");
    lcd.setCursor(0,2); lcd.print("VOLVIENDO AL MENU...");
    lcd.setCursor(0,3); lcd.print("********************");
    delay(3000);
  }
}





































void leetablero(String entrada) {
  // 1. Buscamos los delimitadores { }
  int inicio = entrada.indexOf('{');
  int fin = entrada.indexOf('}');

  // Si el formato es incorrecto, salimos para evitar errores
  if (inicio == -1 || fin == -1) {
    Serial.println("Error: Formato de string no valido");
    return;
  }

  // 2. Extraemos el contenido
  String contenido = entrada.substring(inicio + 1, fin);
  int fila = 0;
  int col = 0;

  // 3. Recorremos el contenido para llenar la matriz global 'tablero'
  for (int i = 0; i < contenido.length(); i++) {
    char c = contenido.charAt(i);

    if (c == ',') {
      col++;  // Siguiente columna
    } else if (c == ';') {
      fila++;   // Siguiente fila
      col = 0;  // Reiniciamos columna
    } else if (c >= '0' && c <= '2') {
      // Verificamos que no nos salgamos del 3x3
      if (fila < 3 && col < 3) {
        // CONVERSIÓN: Restamos '0' para obtener el INT real
        tablero[fila][col] = c - '0';
      }
    }
  }
}

void actualizarLCD() {

  
  // Dibuja el tablero centrado
  for (int i = 0; i < 3; i++) {
    lcd.setCursor(7, i + 1); 
    for (int j = 0; j < 3; j++) {
      if (tablero[i][j] == 0) lcd.print("."); 
      else if (tablero[i][j] == 1) lcd.print("X");
      else if (tablero[i][j] == 2) lcd.print("O");
      if (j < 2) lcd.print("|"); 
    }
  }
}

// --- FUNCIÓN PARA LIMPIAR LA MATRIZ ---
void inicializarTablero() {
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      tablero[i][j] = 0; // Seteamos cada celda a cero
    }
  }
  
}


void MensajeInicial(){
  lcd.clear();
  lcd.setCursor(5,0);
  lcd.print("JUEGO DEL");
  lcd.setCursor(4,2);
  lcd.print("TRES EN RAYA ");
  delay(1500);



}








int comprobarGanador() {
  // 1. Comprobar Filas
  for (int i = 0; i < 3; i++) {
    if (tablero[i][0] != 0 && tablero[i][0] == tablero[i][1] && tablero[i][1] == tablero[i][2]) {
      return tablero[i][0];
    }
  }

  // 2. Comprobar Columnas
  for (int i = 0; i < 3; i++) {
    if (tablero[0][i] != 0 && tablero[0][i] == tablero[1][i] && tablero[1][i] == tablero[2][i]) {
      return tablero[0][i];
    }
  }

  // 3. Comprobar Diagonal Principal (\)
  if (tablero[0][0] != 0 && tablero[0][0] == tablero[1][1] && tablero[1][1] == tablero[2][2]) {
    return tablero[0][0];
  }

  // 4. Comprobar Diagonal Inversa (/)
  if (tablero[0][2] != 0 && tablero[0][2] == tablero[1][1] && tablero[1][1] == tablero[2][0]) {
    return tablero[0][2];
  }

  // --- NUEVA LÓGICA: Comprobar si está lleno ---
  bool hayEspacioVacio = false;
  for (int f = 0; f < 3; f++) {
    for (int c = 0; c < 3; c++) {
      if (tablero[f][c] == 0) {
        hayEspacioVacio = true;  // Todavía se puede jugar
        break;
      }
    }
  }

  if (!hayEspacioVacio) {
    return 3;  // Código para EMPATE
  }

  return 0;  // El juego sigue
}






void imprimirTablero() {
  Serial.println("--- Estado del Tablero (3x3) ---");

  for (int f = 0; f < 3; f++) {
    Serial.print("| ");  // Borde izquierdo
    for (int c = 0; c < 3; c++) {
      Serial.print(tablero[f][c]);
      Serial.print(" ");  // Espacio entre números
    }
    Serial.println("|");  // Borde derecho y salto de línea
  }

  Serial.println("--------------------------------");
}




void vaciarTablero() {
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      tablero[i][j] = 0; // Seteamos cada celda a cero
    }
  }
}

