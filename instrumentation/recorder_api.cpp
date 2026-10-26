// Disk-state extension for the existing vanilla emulator. MMC1 + CHR ROM only.
// No C++ object images, pointers, callbacks, or training interventions are saved.
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include "nes_emu/emulator.hpp"
#include "nes_emu/mappers/mapper_SxROM.hpp"
#include "trace_hooks.hpp"
namespace NES {
bool debug_trace_enabled=false;
void debug_instruction(CPU&,MainBus&){}
void debug_access(char,NES_Address,NES_Byte){}
void debug_rom_read(NES_Address,NES_Byte,const NES_Byte*){}
struct RecorderState {
    std::vector<unsigned char> bytes;
    size_t cursor=0;
    bool reading=false;
    template<class T> void operator()(T& value) {
        static_assert(std::is_trivially_copyable<T>::value,"state field must contain no pointers");
        if(reading) {
            if(cursor+sizeof(T)>bytes.size()) throw std::runtime_error("short state");
            std::memcpy(&value,bytes.data()+cursor,sizeof(T)); cursor+=sizeof(T);
        } else {
            auto start=reinterpret_cast<unsigned char*>(&value);
            bytes.insert(bytes.end(),start,start+sizeof(T));
        }
    }
    void transfer(Emulator& e) {
        auto& io=*this;
        if(e.get_mapper_number()!=1 || e.has_chr_ram()) throw std::runtime_error("requires MMC1 CHR ROM");
        auto& m=*dynamic_cast<MapperSxROM*>(e.mapper.get());
        auto b=e.bus.save_state(); io(b.ram);
        auto v=e.picture_bus.save_state(); io(v.ram); io(v.palette);
        // These are mutable emulator state, missing from its original Snapshot.
        io(e.picture_bus.name_tables); io(e.picture_bus.write_generation);
        io(e.cpu.register_PC); io(e.cpu.register_SP); io(e.cpu.register_A);
        io(e.cpu.register_X); io(e.cpu.register_Y); io(e.cpu.flags.byte);
        io(e.cpu.skip_cycles); io(e.cpu.cycles);
        for(auto& c:e.controllers) {io(c.is_strobe); io(c.joypad_buttons); io(c.joypad_bits);}
        io(m.mode_chr); io(m.mode_prg); io(m.temp_register); io(m.write_counter);
        io(m.register_prg); io(m.register_chr0); io(m.register_chr1);
        io(m.prg_ram_writable); io(m.name_table_mirroring); io(m.name_table_mirroring_changed);
        // Fixed by this ROM, so no untrusted length is allocated on load.
        for(auto& value:m.prg_ram) io(value);
        auto p=e.ppu.save_state();
/*PPU_FIELDS*/
        if(reading) {
            if(cursor!=bytes.size()) throw std::runtime_error("state size mismatch");
            e.bus.load_state(b);
            // load_state increments a cache generation: preserve the saved value.
            auto generation=e.picture_bus.write_generation;
            auto tables=e.picture_bus.name_tables;
            auto mirroring_changed=m.name_table_mirroring_changed;
            e.picture_bus.load_state(v); e.picture_bus.write_generation=generation;
            m.calculatePRGWindows(); m.calculateCHRWindows();
            e.wire_mapper();
            e.picture_bus.write_generation=generation;
            e.picture_bus.name_tables=tables;
            m.name_table_mirroring_changed=mirroring_changed;
            e.ppu.load_state(p);
        }
    }
};
}
using NES::Emulator;
extern "C" {
__declspec(dllexport) const char* rec_build(){return "RECORDER_BUILD_ID";}
__declspec(dllexport) void* rec_create(const char* path){try{return new Emulator(path);}catch(...){return nullptr;}}
__declspec(dllexport) void rec_destroy(Emulator* e){delete e;}
__declspec(dllexport) void rec_reset(Emulator* e){e->reset();}
__declspec(dllexport) unsigned char* rec_ram(Emulator* e){return e->get_memory_buffer();}
__declspec(dllexport) unsigned int* rec_screen(Emulator* e){return e->get_screen_buffer();}
__declspec(dllexport) int rec_step(Emulator* e,int mask){try{*e->get_controller(0)=mask;e->step();return 0;}catch(...){return -1;}}
__declspec(dllexport) int rec_save(Emulator* e,unsigned char* out,int capacity){
    try{NES::RecorderState s;s.transfer(*e);if(capacity<int(s.bytes.size()))return -1;
        std::memcpy(out,s.bytes.data(),s.bytes.size());return int(s.bytes.size());}catch(...){return -1;}}
__declspec(dllexport) int rec_load(Emulator* e,const unsigned char* data,int length){
    try{NES::RecorderState s;s.reading=true;s.bytes.assign(data,data+length);s.transfer(*e);return 0;}catch(...){return -1;}}
}
