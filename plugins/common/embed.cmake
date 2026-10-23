# cmake -DINPUT=file -DOUTPUT=header -DNAME=symbol -P embed.cmake
# Writes a zero-terminated byte array holding INPUT and `const char* NAME` to it.
file(READ "${INPUT}" hex HEX)
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," bytes "${hex}")
file(WRITE "${OUTPUT}"
    "// Generated from ${INPUT}; do not edit.\n#pragma once\n"
    "static const unsigned char ${NAME}_bytes[] = {${bytes}0x00};\n"
    "static const char* const ${NAME} = reinterpret_cast<const char*>(${NAME}_bytes);\n")
