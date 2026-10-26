#pragma once
#include "nes_emu/cpu.hpp"
namespace NES {
extern bool debug_trace_enabled;
void debug_instruction(CPU& cpu, MainBus& bus);
void debug_access(char kind, NES_Address address, NES_Byte value);
void debug_rom_read(NES_Address address, NES_Byte value, const NES_Byte* pointer);
}
