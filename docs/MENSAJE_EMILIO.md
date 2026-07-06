# Mensaje para Emilio (listo para WhatsApp)

Qué onda Emilio! Vi tu mensaje — el md no aparecía porque lo local estaba 36
commits atrás; ya jalamos todo tu trabajo (`acb2ba3`, brutal el sprint 👏).
Trabajamos toda la noche encima de tu rama en `luis/clean-room-paper-v1` y te
explico las dudas + lo que ya quedó:

**¿Por qué "cosas que tiene que hacer un humano"?** Son 4, y ya hicimos 3:

1. **Clean-room rerun (HECHO ✅)** — por eso tenía que ser "otra persona/otra
   máquina": para separar tu resultado de tu entorno. Re-corrimos TODO desde
   cero en la M3 Max de Luis (Python 3.13, torch 2.12, CPU): bajar los 10
   partidos, verificar el split (hash idéntico `0d66a904…`), reconstruir
   ventanas (315,400, y ojo: con el fix de periodos salen los DOS tiempos —
   tu artefacto viejo period-1-only era metadata stale, no data faltante),
   entrenar seeds 7/11/23, y correr todos los gates. **Resultado: se
   reproduce TODO, blocker por blocker.** Falsificación `controls_passed`
   (y además la corrimos en TEST held-out: también pasa, o sea no es artefacto
   de selección), probes bloquean en tus mismos 2 targets (global-x y
   team-shape sin normalizar), los 4 blockers de discovery idénticos, overall
   `blocked`. Tu resultado es real y reproducible.

2. **Anotación ciega (PANEL DIAGNÓSTICO HECHO ✅ / humana pendiente)** — tú no
   puedes ser el anotador porque ya viste las llaves privadas. Regeneramos el
   paquete balanceado (20+20, 40 GIFs, validator passed) y lo anotaron A
   CIEGAS 5 modelos frontera en contextos limpios (Fable, Opus, Sonnet,
   Codex GPT-5.5 y Kimi — 40/40 cada uno).
   Resultado durísimo y útil: **enriquecimiento NEGATIVO** — los clips de
   residual alto salieron "tácticos" solo 5% vs 47% de los controles ocultos
   (RD −0.42, Fisher p=1.0), y el label dominante fue `tracking_artifact`.
   O sea: el residual detecta tracking roto, no táctica — confirmación ciega
   de tu regla "no llamarlo tactical surprise". Acuerdo entre modelos:
   Fleiss κ=0.33; los más alineados fable↔opus κ=0.80 y codex↔opus κ=0.77. Tu pasada humana sigue siendo la
   compuerta oficial: 40 GIFs, ~30-40 min, guía en
   `docs/BLINDED_ANNOTATION_GUIDE.md`.

3. **Decisión rediseñar-o-aceptar (TOMADA ✅, revisable el lunes)** — con los
   gates bloqueados y reproducidos, la tabla de PAPER_FINAL_PATH manda a
   "resultado negativo honesto + protocolo como contribución". Bonus técnico
   para el rediseño: hicimos la ablación sin reconstrucción → los controles
   temporales SUBEN (shuffled 7.6×, no-motion 48×) pero los de identidad se
   caen a ~1.0. O sea: la reconstrucción compra sensibilidad de identidad a
   costa de direccionalidad temporal. Ahí está la pista de qué atacar.

4. **Incertidumbre (HECHO ✅)** — seed-level y match-level en todos los
   summaries y en el paper.

**El paper YA EXISTE y compila** 📄: "Gates Before Claims: A Leakage-Controlled
Evaluation Protocol and a Blocked Case Study for Self-Supervised
Soccer-Tracking Representations" — 15 págs estilo NeurIPS (preprint), autores
Luis (Tec de Monterrey) y tú (UC Berkeley). Cero números a mano: todo se
genera de los artefactos con `scripts/make_paper_assets.py`. Ya lo pasamos por
3 revisores adversariales (area-chair NeurIPS, revisor JEPA estilo LeCun, y un
auditor de reproducibilidad claim-por-claim) y aplicamos todo. Nota: le
cambiamos el nombre al modelo en el paper a "TD hybrid" porque "TD-JEPA" ya
existe en la literatura de RL zero-shot.

**Para el lunes:** (1) tu anotación humana si quieres que entre al paper,
(2) revisar PDF y decidir venue (los revisores sugieren workshop
reproducibilidad/sports-ML ya, o Datasets&Benchmarks si añadimos el
experimento de "leak plantado" con el pipeline legacy — está anotado como next
step), (3) push/merge de `luis/clean-room-paper-v1` (todo está local, 181
tests verdes, ruff limpio).

Detalle completo en `docs/CLEAN_ROOM_REPORT.md`. Nos vemos el lunes 🐻
