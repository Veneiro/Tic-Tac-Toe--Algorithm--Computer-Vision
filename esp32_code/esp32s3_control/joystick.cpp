#include "joystick.h"

/** @brief Constructor: inicializa todos los números de pin, los límites del rango ADC, la zona muerta y el estado del antirrebote.
 *  @param pinJoyX      Pin GPIO conectado a la salida ADC del eje X.
 *  @param pinJoyY      Pin GPIO conectado a la salida ADC del eje Y.
 *  @param pinJoyButton Pin GPIO conectado al botón del joystick (activo en bajo con PULLUP).
 *  @param joyXMin      Lectura ADC mínima en bruto en el eje X.
 *  @param joyXCenter   Lectura ADC central (en reposo) en el eje X.
 *  @param joyXMax      Lectura ADC máxima en bruto en el eje X.
 *  @param joyYMin      Lectura ADC mínima en bruto en el eje Y.
 *  @param joyYCenter   Lectura ADC central (en reposo) en el eje Y.
 *  @param joyYMax      Lectura ADC máxima en bruto en el eje Y. */
Joystick::Joystick(int pinJoyX,
                   int pinJoyY,
                   int pinJoyButton,
                   int joyXMin,
                   int joyXCenter,
                   int joyXMax,
                   int joyYMin,
                   int joyYCenter,
                   int joyYMax)
{
    pinX = pinJoyX;
    pinY = pinJoyY;
    pinButton = pinJoyButton;

    valorX = joyXCenter;
    valorY = joyYCenter;
    valorZ = joyYCenter;

    xMin = joyXMin;
    xCenter = joyXCenter;
    xMax = joyXMax;

    yMin = joyYMin;
    yCenter = joyYCenter;
    yMax = joyYMax;

    deadZone = 80;

    modoZ = false;

    lastButtonReading = HIGH;
    buttonState = HIGH;
    lastDebounceTime = 0;
    debounceDelay = 50;
}

/** @brief Configura los pines GPIO del joystick y establece el ADC a resolución de 12 bits. */
void Joystick::begin()
{
    pinMode(pinX, INPUT);
    pinMode(pinY, INPUT);
    pinMode(pinButton, INPUT_PULLUP);

    analogReadResolution(12);
}

/** @brief Lee los valores ADC y actualiza valorX, valorY y valorZ según el modo actual.
 *  En modo XY la salida del eje Y controla Z y los ejes XY se centran; en modo normal ocurre lo contrario.
 *  También llama a updateButtonToggle() para detectar cambios de modo. */
void Joystick::update()
{
    updateButtonToggle();

    int lecturaX = analogRead(pinX);
    int lecturaY = analogRead(pinY);

    if (modoZ)
    {
        // Modo Z:
        // el joystick Y controla Z,
        // mientras X e Y se consideran centrados.
        valorX = xCenter;
        valorY = yCenter;
        valorZ = lecturaY;
    }
    else
    {
        // Modo XY:
        // el joystick controla X e Y,
        // mientras Z se considera centrado.
        valorX = lecturaX;
        valorY = lecturaY;
        valorZ = yCenter;
    }
}

/** @brief Lector de botón con antirrebote que alterna modoZ en cada pulsación confirmada.
 *  Usa lógica INPUT_PULLUP: LOW = pulsado. El cambio solo se activa en el flanco de bajada
 *  una vez transcurrido el retardo de antirrebote. */
void Joystick::updateButtonToggle()
{
    bool reading = digitalRead(pinButton);

    if (reading != lastButtonReading)
    {
        lastDebounceTime = millis();
    }

    if ((millis() - lastDebounceTime) > debounceDelay)
    {
        if (reading != buttonState)
        {
            buttonState = reading;

            // Con INPUT_PULLUP:
            // suelto  -> HIGH
            // pulsado -> LOW
            if (buttonState == LOW)
            {
                modoZ = !modoZ;
            }
        }
    }

    lastButtonReading = reading;
}

/** @brief Mapea un valor ADC en bruto a una velocidad en [-Vmax, Vmax] con zona muerta.
 *  Los valores dentro de la deadZone del centro devuelven 0. Fuera de la zona muerta, la salida se
 *  escala linealmente usando el semirango apropiado.
 *  @param value           Lectura ADC en bruto a mapear.
 *  @param center          Valor ADC correspondiente a la posición neutral (en reposo).
 *  @param minVal          Valor ADC mínimo posible en este eje.
 *  @param maxVal          Valor ADC máximo posible en este eje.
 *  @param Vmax            Magnitud máxima de velocidad de salida.
 *  @param invertDirection Si es verdadero, el signo de la salida se invierte.
 *  @return Velocidad mapeada en el rango [-Vmax, Vmax]. */
float Joystick::mapToVelocity(int value,
                              int center,
                              int minVal,
                              int maxVal,
                              float Vmax,
                              bool invertDirection)
{
    int delta = value - center;

    if (abs(delta) < deadZone)
    {
        return 0.0f;
    }

    float v;

    if (delta > 0)
    {
        v = ((float)(value - center) / (float)(maxVal - center)) * Vmax;
    }
    else
    {
        v = ((float)(value - center) / (float)(center - minVal)) * Vmax;
    }

    if (invertDirection)
    {
        v = -v;
    }

    v = constrain(v, -Vmax, Vmax);

    return v;
}

/** @brief Devuelve el último valor ADC en bruto muestreado del eje X.
 *  @return Lectura ADC en bruto (0–4095 a resolución de 12 bits). */
int Joystick::getValorX() const
{
    return valorX;
}

/** @brief Devuelve el último valor ADC en bruto muestreado del eje Y.
 *  @return Lectura ADC en bruto (0–4095 a resolución de 12 bits). */
int Joystick::getValorY() const
{
    return valorY;
}

/** @brief Devuelve el último valor ADC en bruto muestreado del eje Z.
 *  @return Lectura ADC en bruto (0–4095 a resolución de 12 bits). */
int Joystick::getValorZ() const
{
    return valorZ;
}

/** @brief Devuelve la velocidad mapeada del eje X en [-Vmax, Vmax].
 *  @param Vmax Magnitud máxima de velocidad.
 *  @return Velocidad en X; 0 dentro de la zona muerta. */
float Joystick::getVx(float Vmax)
{
    return mapToVelocity(valorX, xCenter, xMin, xMax, Vmax, false);
}

/** @brief Devuelve la velocidad mapeada del eje Y en [-Vmax, Vmax].
 *  @param Vmax Magnitud máxima de velocidad.
 *  @return Velocidad en Y; 0 dentro de la zona muerta. */
float Joystick::getVy(float Vmax)
{
    return mapToVelocity(valorY, yCenter, yMin, yMax, Vmax, false);
}

/** @brief Devuelve la velocidad mapeada del eje Z en [-Vmax, Vmax] con la dirección invertida.
 *  @param Vmax Magnitud máxima de velocidad.
 *  @return Velocidad en Z; 0 dentro de la zona muerta. */
float Joystick::getVz(float Vmax)
{
    return mapToVelocity(valorZ, yCenter, yMin, yMax, Vmax, true);
}

/** @brief Devuelve el estado actual del modo Z.
 *  @return Verdadero si el joystick está en modo de control Z; falso para modo XY. */
bool Joystick::isModoZ() const
{
    return modoZ;
}

/** @brief Establece el umbral de zona muerta para todos los ejes.
 *  @param dz Radio en cuentas ADC alrededor del centro que se mapea a velocidad cero. */
void Joystick::setDeadZone(int dz)
{
    deadZone = dz;
}

/** @brief Establece el retardo de antirrebote utilizado por el lector de alternancia del botón.
 *  @param delayMs Tiempo mínimo estable en milisegundos antes de que se acepte un flanco del botón. */
void Joystick::setDebounceDelay(unsigned long delayMs)
{
    debounceDelay = delayMs;
}