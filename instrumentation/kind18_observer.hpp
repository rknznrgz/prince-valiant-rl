// Experimental combat observer for the verified kind18 and kind14 paths.
// Included only by the sidecar API TU.
#include <cstdint>

namespace NES {
struct Kind18Event {
    std::uint64_t native_frame;
    std::uint32_t target_slot, projectile_slot, enemy_kind;
    std::uint32_t handler_bank, commit_pc, lethal_pc;
    std::uint32_t event_type, counter_before, counter_after;
};
static_assert(sizeof(Kind18Event)==48,"observer ABI v2 event size");
enum CombatEventType { ACCEPTED_HIT=1, DEFEAT=2 };

struct Kind18Observer {
    enum Stage { NONE, DISPATCH, HANDLER, WRITE_SEEN, COMMITTED, ZERO_BRANCH,
        KIND14_DISPATCH, KIND14_HANDLER, KIND14_WRITE_SEEN, KIND14_ZERO_BRANCH,
        KIND14_GATE_WRITTEN, KIND14_PHASE_WRITTEN, KIND14_KIND_WRITTEN };
    Stage stage=NONE;
    unsigned target=0, projectile=0;
    unsigned counter_upper=0;
    unsigned counter_before=0, counter_after=0;
    unsigned enemy_kind=0;
    unsigned capacity=2, count=0;
    bool overflow=false;
    std::uint64_t frame=0;
    Kind18Event events[2]{};
    const NES_Byte* prg=nullptr;
    NES_Byte* ram=nullptr;
    unsigned last_pc=0, last_x=0;
    long last_offset=-1;

    void clear() {
        stage=NONE;count=0;overflow=false;frame=0;events[0]={};events[1]={};
        last_pc=0;last_offset=-1;ram=nullptr;counter_upper=0;
        counter_before=counter_after=enemy_kind=0;
    }
    bool at(unsigned bank,unsigned pc) const {
        return last_pc==pc && last_offset==long(bank*16384+(pc&0x3fff));
    }
    static unsigned word(const NES_Byte* r,unsigned a) {return r[a]+256u*r[a+1];}
    static bool lethal_counter_commit(unsigned before,unsigned after) {
        return (before&0x1f)==1 && (after&0x1f)==0
            && (before&0xe0)==(after&0xe0);
    }
    bool scene() const {
        return ram[0x33]==8 && ram[0x4D]==2 && ram[0x2F4]==0
            && word(ram,0x50E)==0x9F81 && ram[0x60]==0 && ram[0xF7]<128;
    }
    bool final_scene() const {
        return ram[0x33]==7 && ram[0x4D]==2 && ram[0x2F4]==0
            && ram[0x60]==0 && ram[0xF7]<128;
    }
    void emit(unsigned type,unsigned kind,unsigned commit,unsigned lethal,
        unsigned before,unsigned after) {
        if(count<capacity) {
            events[count++]={frame,target,projectile,kind,2,commit,lethal,
                type,before,after};
        } else overflow=true;
    }
    void instruction(CPU& cpu,MainBus& bus) {
        const auto r=cpu.save_state();
        last_pc=r.register_PC;last_x=r.register_X;
        last_offset=bus.debug_prg_offset(r.register_PC,prg);
        ram=bus.get_memory_buffer();
        if(at(7,0xDB98)) {
            stage=NONE;
            const unsigned s=ram[0x41], p=ram[0x0501];
            const unsigned kind=s<10?ram[0xC1+s]:0;
            const bool attribution=s<10 && r.register_X==s && ram[0x42]==2*s
                && r.register_Y==2*s && p==4 && ram[0x50D]==1
                && kind==18 && ram[0x99+s]==0
                && (ram[0xCB+s]==5 || ram[0xCB+s]==7)
                && (ram[0xB7+s]&0x1f)==1;
            if(attribution && scene() && word(ram,0x1B)==0x87E2
                && bus.debug_prg_offset(0x87E2,prg)==long(2*16384+0x7E2)) {
                target=s;projectile=p/2;enemy_kind=18;
                counter_upper=ram[0xB7+s]&0xe0;
                counter_before=counter_after=ram[0xB7+s];stage=DISPATCH;
            } else if(s<10 && r.register_X==s && ram[0x42]==2*s
                && r.register_Y==2*s && (p==0 || p==2 || p==4 || p==6)
                && (ram[0x0509+p]&0x7f)==1 && kind==14
                && ram[0x99+s]==1 && ram[0xB7+s]>0 && final_scene()
                && word(ram,0x1B)==0x8D23
                && bus.debug_prg_offset(0x8D23,prg)==long(2*16384+(0x8D23&0x3fff))) {
                target=s;projectile=p/2;enemy_kind=14;
                counter_before=ram[0xB7+s];counter_after=counter_before;
                stage=KIND14_DISPATCH;
            }
        } else if(at(2,0x87E2)) {
            stage=(stage==DISPATCH && last_x==target)?HANDLER:NONE;
        } else if(at(2,0x87F8)) {
            // The write callback runs BEFORE the bus writes RAM. Only this
            // following executed instruction confirms that the write committed.
            const unsigned counter=ram[0xB7+target];
            stage=(stage==WRITE_SEEN && last_x==target
                && (counter&0x1f)==0 && (counter&0xe0)==counter_upper)
                ?COMMITTED:NONE;
        } else if(at(2,0x87FA)) {
            stage=(stage==COMMITTED && last_x==target && (r.flags&2))?ZERO_BRANCH:NONE;
        } else if(at(2,0x87FC)) {
            if(stage==ZERO_BRANCH && last_x==target && ram[0xC1+target]==18
                && (ram[0xB7+target]&0x1f)==0
                && (ram[0xB7+target]&0xe0)==counter_upper
                && ram[0x50D]==2 && scene()) {
                emit(DEFEAT,18,0x87F6,0x87FC,counter_before,counter_after);
            }
            stage=NONE; // one event per observed lethal branch
        } else if(at(2,0x8D23)) {
            stage=(stage==KIND14_DISPATCH && last_x==target
                && ram[0xC1+target]==14 && ram[0x99+target]==1
                && ram[0xB7+target]==counter_before)?KIND14_HANDLER:NONE;
        } else if(at(2,0x8D27)) {
            if(stage==KIND14_WRITE_SEEN && last_x==target && (r.flags&2)
                && counter_after==0 && ram[0xB7+target]==counter_after) {
                emit(ACCEPTED_HIT,14,0x8D23,0,counter_before,counter_after);
                stage=KIND14_ZERO_BRANCH;
            } else stage=NONE;
        } else if(at(2,0x8D3F)) {
            if(stage==KIND14_WRITE_SEEN && last_x==target && !(r.flags&2)
                && counter_after>0 && ram[0xB7+target]==counter_after) {
                emit(ACCEPTED_HIT,14,0x8D23,0,counter_before,counter_after);
            }
            stage=NONE; // accepted nonlethal decrement; no defeat event
        } else if(at(2,0x8D34)) {
            if(stage==KIND14_KIND_WRITTEN && last_x==target
                && ram[0x204]==1 && ram[0x99+target]==0
                && ram[0xC1+target]==13 && ram[0xB7+target]==0
                && final_scene()) emit(DEFEAT,14,0x8D23,0x8D27,
                    counter_before,counter_after);
            stage=NONE;
        } else if(at(7,0xDB9B)) stage=NONE; // handler returned without qualification
    }
    void access(char kind,NES_Address address,NES_Byte value) {
        const unsigned before=ram[0xB7+target];
        if(kind=='W' && stage==HANDLER && at(2,0x87F6) && last_x==target
            && address==0xB7+target && (before&0xe0)==counter_upper
            && lethal_counter_commit(before,value)) {
            counter_before=before;counter_after=value;
            stage=WRITE_SEEN;
        }
        else if(kind=='W' && stage==KIND14_HANDLER && at(2,0x8D23)
            && last_x==target && address==0xB7+target
            && before==counter_before && before>0 && value==before-1) {
            counter_after=value;
            stage=KIND14_WRITE_SEEN;
        }
        else if(kind=='W' && stage==KIND14_ZERO_BRANCH && at(2,0x8D29)
            && address==0x204 && value==1)
            stage=KIND14_GATE_WRITTEN;
        else if(kind=='W' && stage==KIND14_GATE_WRITTEN && at(2,0x8D2E)
            && last_x==target && address==0x99+target && value==0)
            stage=KIND14_PHASE_WRITTEN;
        else if(kind=='W' && stage==KIND14_PHASE_WRITTEN && at(2,0x8D32)
            && last_x==target && address==0xC1+target && value==13)
            stage=KIND14_KIND_WRITTEN;
    }
};

// All durable observation state belongs to the owner. TLS only routes the
// currently executing rec_step into its owner, including multiple DLL owners.
thread_local bool debug_trace_enabled=false;
static thread_local Kind18Observer* active_observer=nullptr;
void debug_instruction(CPU& cpu,MainBus& bus) {
    if(active_observer)active_observer->instruction(cpu,bus);
}
void debug_access(char kind,NES_Address address,NES_Byte value) {
    if(active_observer)active_observer->access(kind,address,value);
}
void debug_rom_read(NES_Address,NES_Byte,const NES_Byte*) {}

struct ObserverScope {
    Kind18Observer* previous;
    bool previous_enabled;
    explicit ObserverScope(Kind18Observer& observer)
        :previous(active_observer),previous_enabled(debug_trace_enabled) {
        active_observer=&observer;debug_trace_enabled=true;
    }
    ~ObserverScope(){active_observer=previous;debug_trace_enabled=previous_enabled;}
};
}
