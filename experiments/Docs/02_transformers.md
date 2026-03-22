# Experimento 3: ir a LSA-T con modelos más grandes

Ya es claro que necesito un modelo gloss-free o semi-gloss-free.

## Idea: usar LSA-T + segmentación débil
Para tener un baseline, mi idea es:
- Usar LSA-T (videos continuos con subtítulos en español).
- Convertir subtítulos → glosas usando NLP.
- Usar esas glosas para **segmentación débil**:
  - no hay alineación perfecta,
  - pero sí una secuencia aproximada de glosas.

Esto permite entrenar un modelo que:

- entiende estructura temporal,
- no depende de ventanas artificiales,
- no exige anotación exacta.

### Cambios sobre la arquitectura original de LSA-T

El paper original usa:

- 3 convoluciones temporales con kernel tamaño 1 → **solo 3 features por frame**  

Siento que es demasiada reducción, para eso quiero usar **~50 features** temporales en vez de 3, o pasar los landmarks directamente al modelo secuencial sin convoluciones.

Tampoco me convence pasarle los landmarks faciales crudos, ya que estaria exigiendole al modelo que tambien aprenda reconocimiento de emociones
Mi nueva propuesta:
- Un modelo **ya entrenado** o una RNN/transformer dedicado SOLO a interpretar la cara y pasar la **representación embebida** (no raw) al modelo principal.

Otra cosa que se me ocurre es hacer algun entrenamiento inicial, y luego entrenar a un transformer con una RNN dedicada a recoenocer glosas individuales y con 2 neuronas sigmoides dedicadas a decir si
- la salida actual del transformer se lo meto a la rnn
- deberia tomar lo que esta dando la rnn como seña

y a la hora de entrenar, no modificar los parametros de la rnn. La idea con eso es que el transformer aprenda a segmentar las señas, y lo bueno de eso es que seria mas facil enseñarle nuevas señas ya que solo habria que meterle un par de ejemplos grabados a mano a la rnn.

Aun asi no funcionaria porque no importa quien las recorte(ad hoc o transformer), los problemas seguirian siendo los mismos(co-articulacion, diferentes velocidades, seña aislada vs en conjunto, etc)
## Obtencion de datos
- [CN Sordos](https://www.youtube.com/@CNSORDOSARGENTINA/videos)
- [Canal de asociacion civil](https://www.youtube.com/c/CanalesAsociaci%C3%B3nCivil/videos)
- [Videolibros en señas](https://www.videolibros.org/video/106)

## Resultados
