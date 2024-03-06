# SmartDots (IN DEVELOPMENT ....)
Este proyecto pretende encontrar una posible solución a un **problema de búsqueda** en un entorno dado con dos jugadores utilizando **Deep Reinforcement Learning**. Además, luego de encontrar un jugador al otro la tarea cambia a **mantener en visibilidad el mayor tiempo posible**.

---

## Descripción del entorno
El programa está diseñado en Pygame y consta de las siguientes características:
| Características | Descripción |
| :-- | :-- |
| Dimensiondes del entorno | 200x200 px |
| Obstáculos | Paredes en los bordes del entorno con ancho de 20px |
| Jugadores | Discos con radio de 8px (azul &rarr; perseguidor, rojo &rarr; evasor ) |

<div align="center"><img src="result images/game environment.png" alt="Entorno"/></div>

---

## Descripción de los jugadores

<table>  
    <tr>
      <th>Características</th>
      <th>Descripción</th>
    </tr>
    <tr>
      <td>Acciones</td>
      <td>
        <ul>
          <li>"NO ACTION": No desplazarse</li>
          <li>"LEFT": Desplazamiento de 2px hacia la izquierda</li>
          <li>"UP": Desplazamiento de 2px hacia arriba</li>
          <li>"RIGHT": Desplazamiento de 2px hacia la derecha</li>
          <li>"DOWN": Desplazamiento de 2px hacia abajo</li>
          <li>"DOUBLE-LEFT": Desplazamiento de 4px hacia la izquierda</li>
          <li>"DOUBLE-UP": Desplazamiento de 4px hacia arriba</li>
          <li>"DOUBLE-RIGHT": Desplazamiento de 4px hacia la derecha</li>
          <li>"DOUBLE-DOWN": Desplazamiento de 4px hacia abajo</li>          
        </ul>
      </td>
    </tr>
  <tr>
    <td>Sensores</td>
    <td>
      <ul>
        <li>Cámara panorámica del entorno</li>
        <li>Láser desde la posición del jugador hacia las 4 direcciones hacia donde se puede desplazar</li>
      </ul>        
    </td>
  </tr>
</table>


