# Referencias de terceros

Esta entrega contiene código, configuración de ejemplo y manifiestos, no ruedas, DLL, instaladores ni pesos de terceros. El instalador obtiene las versiones fijadas de PyPI; la utilidad de modelo descarga de su proveedor solo por orden explícita.

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper/blob/master/LICENSE): MIT.
- [CTranslate2](https://github.com/OpenNMT/CTranslate2/blob/master/LICENSE): MIT.
- [Modelo convertido large-v3](https://huggingface.co/Systran/faster-whisper-large-v3): ficha con licencia MIT; conservar atribuciones del proveedor si se redistribuye por separado.
- [cuDNN](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/eula.html) y [CUDA](https://docs.nvidia.com/cuda/eula/index.html): condiciones NVIDIA; los paquetes NVIDIA fijados se identifican como propietarios. No se redistribuyen aquí.
- [PyAV](https://github.com/PyAV-Org/PyAV): BSD; sus ruedas incorporan bibliotecas FFmpeg con sus propios términos. [FFmpeg](https://ffmpeg.org/legal.html).
- Otras dependencias fijadas usan, entre otras, MIT, BSD, Apache, MPL y PSF. Sus metadatos y archivos de licencia acompañan las distribuciones originales de PyPI. El lock identifica exactamente los paquetes seleccionados.

Estas referencias son informativas, no una relicencia del conjunto. Antes de redistribuir binarios o pesos debe revisarse su licencia concreta y los avisos incorporados. La licencia del código propio permanece pendiente de elección; no se elimina ni reemplaza ninguna licencia existente.
