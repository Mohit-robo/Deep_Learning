# OneEye — Custom Lib → Container Path Mapping
# DeepStream 7.1 | Base path: /opt/nvidia/deepstream/deepstream-7.1/

> This document maps every file in `libs/` to its **canonical location** inside the
> `deepstream-analytics` container. Use these paths when volume-mounting individual
> files, writing COPY instructions, or running `docker exec` to verify overrides.

---

## DeepStream Root

```
DS_ROOT = /opt/nvidia/deepstream/deepstream-7.1
```

---

## File Mapping Table

| Local file (`libs/`)              | Type          | Container path (absolute)                                                                                         | Notes                                                              |
|-----------------------------------|---------------|-------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|
| `deepstream_source_bin.c`         | C source      | `$DS_ROOT/sources/apps/apps-common/src/deepstream_source_bin.c`                                                   | Source for the common source-bin helper used by sample apps        |
| `eventmsg_payload.cpp`            | C++ source    | `$DS_ROOT/sources/libs/nvmsgconv/eventmsg_payload.cpp`                                                           | Part of the `nvmsgconv` message-conversion library                 |
| `gst-nvdscommonconfig.h`          | C header      | `$DS_ROOT/sources/gst-plugins/gst-nvmultiurisrcbin/gst-nvdscommonconfig.h`                                        | Shared config header for the multi-URI source-bin plugin           |
| `gstdsnvmultiurisrcbin.cpp`       | C++ source    | `$DS_ROOT/sources/gst-plugins/gst-nvmultiurisrcbin/gstdsnvmultiurisrcbin.cpp`                                    | Main implementation of the `nvmultiurisrcbin` GStreamer plugin     |
| `gstdsnvurisrcbin.cpp`            | C++ source    | `$DS_ROOT/sources/gst-plugins/gst-nvurisrcbin/gstdsnvurisrcbin.cpp`                                             | Main implementation of the `nvurisrcbin` GStreamer plugin          |
| `gstdsnvurisrcbin.h`              | C header      | `$DS_ROOT/sources/gst-plugins/gst-nvurisrcbin/gstdsnvurisrcbin.h`                                               | Header for the `nvurisrcbin` plugin                                |
| `nvdsmeta_schema.h`               | C header      | `$DS_ROOT/sources/includes/nvdsmeta_schema.h`                                                                    | DeepStream metadata schema definitions (used across many plugins)  |
| **`libnvds_msgconv.so`**          | Compiled lib  | `$DS_ROOT/lib/libnvds_msgconv.so`                                                                                | Runtime message-conversion library — **override replaces binary**  |
| **`libnvdsgst_nvmultiurisrcbin.so`** | GStreamer plugin | `$DS_ROOT/lib/gst-plugins/libnvdsgst_nvmultiurisrcbin.so`                                                 | GStreamer plugin for multi-source URI bin — **override replaces binary** |
| **`libnvdsgst_nvurisrcbin.so`**   | GStreamer plugin | `$DS_ROOT/lib/gst-plugins/libnvdsgst_nvurisrcbin.so`                                                         | GStreamer plugin for single-source URI bin — **override replaces binary** |

---

## Directory Tree (inside container)

```
/opt/nvidia/deepstream/deepstream-7.1/
├── lib/
│   ├── libnvds_msgconv.so              ← your custom build
│   └── gst-plugins/
│       ├── libnvdsgst_nvmultiurisrcbin.so   ← your custom build
│       └── libnvdsgst_nvurisrcbin.so        ← your custom build
│
└── sources/
    ├── includes/
    │   └── nvdsmeta_schema.h           ← your custom header
    ├── apps/
    │   └── apps-common/
    │       └── src/
    │           └── deepstream_source_bin.c   ← your custom source
    ├── libs/
    │   └── nvmsgconv/
    │       └── eventmsg_payload.cpp    ← your custom source
    └── gst-plugins/
        ├── gst-nvmultiurisrcbin/
        │   ├── gst-nvdscommonconfig.h  ← your custom header
        │   └── gstdsnvmultiurisrcbin.cpp  ← your custom source
        └── gst-nvurisrcbin/
            ├── gstdsnvurisrcbin.h      ← your custom header
            └── gstdsnvurisrcbin.cpp    ← your custom source
```

---

## Key Notes

- **`.so` files** are the only files that take effect at runtime without a rebuild.
  The source (`.c`/`.cpp`) and header (`.h`) files require a recompile inside the
  container or as part of the image build to produce new `.so` artifacts.

- **`libnvds_msgconv.so`** is loaded by `LD_LIBRARY_PATH` from `$DS_ROOT/lib/`.

- **GStreamer plugins** (`.so` under `gst-plugins/`) are discovered via
  `GST_PLUGIN_PATH` which DeepStream sets to `$DS_ROOT/lib/gst-plugins/`.
  Replacing them at this path is sufficient — no GStreamer cache flush needed
  when using a fresh container.

- If you need to **recompile** from the modified sources inside the container:
  ```bash
  # Example for nvurisrcbin
  cd /opt/nvidia/deepstream/deepstream-7.1/sources/gst-plugins/gst-nvurisrcbin
  make && make install
  ```
