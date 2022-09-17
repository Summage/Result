# PA

```
nemu
├── include                    # 存放全局使用的头文件
│   ├── common.h               # 公用的头文件
│   ├── cpu
│   │   ├── decode.h           # 译码相关
│   │   └── exec.h             # 执行相关
│   ├── debug.h                # 一些方便调试用的宏
│   ├── device                 # 设备相关
│   ├── isa                    # ISA相关
│   ├── isa.h                  # ISA相关
│   ├── macro.h                # 一些方便的宏定义
│   ├── memory                 # 访问内存相关
│   ├── monitor
│   │   ├── log.h              # 日志文件相关
│   │   └── monitor.h
│   └── rtl
│       ├── pesudo.h           # RTL伪指令
│       └── rtl.h              # RTL指令相关定义
├── Makefile                   # 指示NEMU的编译和链接
├── Makefile.git               # git版本控制相关
├── runall.sh                  # 一键测试脚本
└── src                        # 源文件
    ├── device                 # 设备相关
    ├── engine
    │   └── interpreter        # 解释器的实现
    ├── isa                    # ISA相关的实现
    │   ├── mips32
    │   ├── riscv32
    │   ├── riscv64
    │   └── x86
    ├── main.c                 # 你知道的...
    ├── memory
    │   └── paddr.c            # 物理内存访问
    └── monitor
        ├── cpu-exec.c         # 指令执行的主循环
        ├── debug              # 简易调试器相关
        │   ├── expr.c         # 表达式求值的实现
        │   ├── log.c          # 日志文件相关
        │   ├── ui.c           # 用户界面相关
        │   └── watchpoint.c   # 监视点的实现
        └── monitor.c
```



## Makefile

### NEMU

```
./tools/qemu-diff/Makefile
./tools/kconfig/Makefile
./tools/spike-diff/Makefile
./tools/difftest.mk
./tools/kvm-diff/Makefile
./tools/fixdep/Makefile
./tools/gen-expr/Makefile
./scripts/config.mk
./scripts/native.mk
./scripts/build.mk
./src/utils/filelist.mk
./src/filelist.mk
./src/isa/filelist.mk
./src/engine/filelist.mk
./src/device/filelist.mk
./Makefile
```

#### 生成.i文件

```makefile
// nemu/scripts/build.mk
MARCRO_OBJS = $(SRCS:%.c=$(OBJ_DIR)/%.i) $(CXXSRC:%.cc=$(OBJ_DIR)/%.i)
$(OBJ_DIR)/%.i:%.c
	@echo + CC $<
	@mkdir -p $(dir $@)
	@$(CC) $(CFLAGS) -E $< | \
		grep -va '^#' | \
		clang-format -  > $(basename $@).i
	@find . -maxdepth 1 -type f -name "*.d"  | xargs rm 

$(OBJ_DIR)/%.i: %.cc
	@echo + CXX $<
	@mkdir -p $(dir $@)
	@$(CXX) $(CFLAGS) $(CXXFLAGS) -E $< | \
		grep -va '^#' | \
		clang-format -  > $(basename $@).i
	@find . -maxdepth 1 -type f -name "*.d"  | xargs rm 
	
expand: $(MACRO_OBJS)
```



## PA0

### 问题记录

`am-kernels test无法通过编译`

> 情况一：Ubuntu21.10使用的glibc版本将宏定义SIGSTACK视作函数调用
>
> ```c 
> //abstract-machine/am/src/native/platform.c
> static void setup_sigaltstack() {
> assert(sizeof(thiscpu->sigstack) == SIGSTKSZ);
>  ....
> }
> //abstract-machine/am/src/native/platform.h
> typedef struct {
>  .....
>  uint8_t sigstack[8192]; //uint8_t sigstack[SIGSTACK]; 
> } __am_cpu_t;
> ```
>
> 情况二:\_SC\_SIGSTKSZ不是8192(本机是12848)
>
> ```c
> # include <unistd.h>
> # define SIGSTKSZ sysconf (_SC_SIGSTKSZ)
> ```

## PA1

> 优雅的退出
>
> 使用make run后直接退出会弹出mk执行报错而直接运行可执行文件不会？
>
> make 在执行结束后会检查shell的自动变量\$?（上一次运行结果）。直接q会返回1，非零，导致报错。直接使用命令行运行虽然不会直接报错，但是echo \$?依然可以看到状态为1。

`init_monitor中全是函数`是为了形成层次化结构,更好的将功能和细节代码(可能依赖于isa或硬件)隔离开来.这种抽象可以让代码模块化方便替换修改.

```c
#include <unistd.h>  
extern char *optarg;  
extern int optind, opterr, optopt;  
#include <getopt.h>
int getopt(int argc, char * const argv[],const char *optstring);  
int getopt_long(int argc, char * const argv[], const char *optstring, const struct option *longopts, int *longindex);  
int getopt_long_only(int argc, char * const argv[], const char *optstring, const struct option *longopts, int *longindex);
const struct option table[] = {
    {"port"     , required_argument, NULL, 'p'},
    {"help"     , no_argument      , NULL, 'h'},//optional_argument
    {0          , 0                , NULL,  0 },
  };//{const char * name; int has_arg; int *flag; int val}
```

> 对于`getopt_long`:
>
> 1) `optstring`为短选项字符串,无:表示只有选项,单:表示必须带参数,双:表示可选参数且**参数(若有)直接与选项相连不插空格
> 2) `longops`为结构体, no/require/optional_argument(0,1,2)对应参数情况(可以用=或空格).`flag`若为空,则选中后getopt_long返回val值,若不为空,则在选中后*\*flag=val*且函数返回0.**最后一个字段必须全0填充**
> 3) `longindex`非空时记录符合的opt下标

> `char * strtok(char * str, const char * delim)`,若传入str非空则开始下一次分割,否则延续上一次的字符串.每次会返回从剩余部分开始到delim处的子字符串,若已经清空则返回NULL.
>
> `char * strtok_r(char * str, const char * delim, char ** saveptr)`相较于前者不同在于,要使用的剩余字符串保存在\*saveptr中,而当str非空时会无视并修改saveptr内容.即通过指定不同的saveptr可以实现对多个字符串的并行解析.

> `输出换行`：换行符是刷新缓冲区的条件之一。

### SDB

### 内存扫描 

小字节序.



### 表达式求值

通过make_token()从表达式中提取关键信息

> 规则为正则表达式和token类组成的二元组,由init_regex编码为reg pattern.

```
<expr> ::= <number>   
  | "(" <expr> ")"    
  | <expr> "+" <expr>  
  | <expr> "-" <expr> 
  | <expr> "*" <expr>
  | <expr> "/" <expr>
  | "+/-" <expr>
```

通过递归将长表达式分拆成按照一定顺序执行的`表达式单元`.除拆除括号外,每一次分拆都需要定位`主运算符`,而这除了优先级外还需要考虑运算符的结合性.

> 考虑到`四则运算等左结合`而`符号等单算符右结合`.对于最低优先级的符号连续出现时,除左一外其余都被视作单目运算符从而获得更高的优先级,所以主操作符只能是左一.而对于不连续的同最低级符号,取最右侧为主运算符,从而自动将最开始就是单目运算符的部分分离开来.
>
> 从上不难看出,更改单目运算符优先级不能在分割中进行,而仅在最终主运算符在最左侧时进行.

#### make_token()

遍历规则匹配,若匹配到当前位置(position)为对象起始则成功识别并用Log()输出成功信息.使用Token结构体(类型+32字节缓冲区)记录信息(tokens, nr_token).(对于部分类型只存类型不够,有些还存在**缓冲区溢出**的可能).

若无法识别则返回position.

#### eval

### 监视点

监视点需要保存被监视`表达式信息`以其`前一次求值的信息`(在设置时进行第一次求值顺便验证表达式正确性).可以选择使用`定长字符数组\变长字符串\token数组`保存信息(最后一种不经由expr而是直接调用eval,效率高但是耦合性略微高于前两者.此处考虑原expr.c中make_token为static,不采取这种做法).

考虑对保存表达式的部分使用`动态内存分配`,因此有必要维护`容量标志`.综合内存利用和运行效率,对于被删除的节点,考虑不释放free_顶层一定个数节点的动态分配内存.`貌似分配的内存最后会被自动释放？添加释放内存函数后却出现了double free`

由于pc并不在普通寄存器之列，要实现对pc的监控需要在cpu_exec.c中添加响应函数。此处pc从0x80000000开始，注意调整。

> ```c
> //static 避免多文件重名导致冲突
> static inline void parse_args(int argc, char *argv[]);//inline 不必要，实际由编译器决定
> ```
>
> 当在多个文件中定义了重名函数，虽然能分别通过编译，但链接会出错。因此，也一般不再头文件定义函数（多重引用会导致重名函数出现）。而`static强调该函数在编译时仅在本文件内可见`，可以避免一部分情况。
>
> 当函数较短且对性能有较大影响，可以使用static inline 修饰后放在头文件。在编译时会额外产生call指令，但在调用check_reg_index(0)编译优化为0开销。

## PA2

[RISCV API](https://nju-projectn.github.io/ics-pa-gitbook/ics2021/nemu-isa-api.html)

### 运行流程

由`cpu_exec(n)`函数模拟cpu运行,识别并处理不同的NEMU_STATE.对于运行中的模拟器.调用`execute(n)`执行指令并记录所用时间.而后更新`NEMU状态`(若结束则打印信息).

`execute`新建一个未初始化的Decode结构体s连同当前pc传入`exec_once`.后者用传入的pc给s的pc,snpc赋值后,调用`isa_exec_once(s)`并将获得的`s->dnpc`作为执行指令后的`cpu.pc`.

`isa_exec_once(s)`从当前地址snpc取值(同时snpc+4)获得指令保存到isa.inst.val.而后通过`decode_exec(s)`译码并执行指令.

```c
typedef struct Decode{
    vaddr_t pc, snpc, dnpc;
	ISADecodeInfo isa;
} Decode;

typedef concat(__GUEST_ISA__, _ISADecodeInfo) ISADecodeInfo;
typedef struct {
    union{uint32_t val;} inst;
} riscv64_ISADecodeInfo;
```

#### 匹配

##### 原理

匹配规则为`二进制编码,name(只是给人看的,没有使用),type(用于识别指令类型),body(展开后为body;语句,需要执行的独特操作)`.

pattern_decode函数将规则转化为3个数字, `shift`对应pat最右侧连续的?个数,`key`在pat为1处取1, `mask`在pat取?处为0.`返回时后两者已经向右位移了shift位`.

##### 代码

```c
// nemu/include/cpu/decode.h
__attribute__((always_inline))
static inline void pattern_decode(const char *str, int len,
    uint64_t *key, uint64_t *mask, uint64_t *shift) {
  uint64_t __key = 0, __mask = 0, __shift = 0;
// 注意str低位对应pat左侧(高位),类似大字节序
// 所以生成mask时str的更高一位填充到mask的低位
#define macro(i) \
  if ((i) >= len) goto finish; \
  else { \
    char c = str[i]; \
    if (c != ' ') { \
      Assert(c == '0' || c == '1' || c == '?', \
          "invalid character '%c' in pattern string", c); \
      __key  = (__key  << 1) | (c == '1' ? 1 : 0); \
      __mask = (__mask << 1) | (c == '?' ? 0 : 1); \
      __shift = (c == '?' ? __shift + 1 : 0); \
    } \
  }
// 通过多层macro嵌套从而使得代码展开后一位对应一个if函数(按照i大小顺序排列),最小段为2
// 当对应数字大于等于len(pat长度-1)时跳转到finish
#define macro2(i)  macro(i);   macro((i) + 1)
......
#define macro64(i) macro32(i); macro32((i) + 32)
  macro64(0);
  // 过长报错
  panic("pattern too long");
#undef macro
finish:
  *key = __key >> __shift;
  *mask = __mask >> __shift;
  *shift = __shift;
}
```

```c
// src/isa/riscv64/inst.c
static int decode_exec(Decode *s) {
  word_t dest = 0, src1 = 0, src2 = 0;
  s->dnpc = s->snpc;
#define INSTPAT_INST(s) ((s)->isa.inst.val)
#define INSTPAT_MATCH(s, name, type, ... /* body */ ) { \
  decode_operand(s, &dest, &src1, &src2, concat(TYPE_, type)); \
  __VA_ARGS__ ; \
}
  INSTPAT_START();
  INSTPAT("??????? ????? ????? ??? ????? 00101 11", auipc  , U, R(dest) = src1 + s->pc); // pat,name,type,执行表达式(展开后会加一个;) 
  // 更多的匹配规则
  INSTPAT_END();
  R(0) = 0; // reset $zero to 0
  return 0;
}
static int decode_exec(Decode *s) {
  word_t dest = 0, src1 = 0, src2 = 0;
  s->dnpc = s->snpc;
  {const void **__instpat_end = &&__instpat_end_;
    ;
    do {
      uint64_t key, mask, shift;
      pattern_decode("??????? ????? ????? ??? ????? 00101 11",
                     (sizeof("??????? ????? ????? ??? ????? 00101 11") - 1),
                     &key, &mask, &shift);
      if (((((s)->isa.inst.val) >> shift) & mask) == key) { // 比对除?外的部分
        {
          decode_operand(s, &dest, &src1, &src2, TYPE_U);
          (cpu.gpr[check_reg_idx(dest)]) = src1 + s->pc;
        };
        goto *(__instpat_end);
      }
    } while (0);
      ......
  __instpat_end_:;
  };
  (cpu.gpr[check_reg_idx(0)]) = 0;
  return 0;
}
```

##### tips

> `do{}while(0)`:使得宏展开获得的代码更加安全(**仅被视为一条指令,后可接;**),减少在不恰当位置展开引起的语法错误.
>
> `__instpat_end_:`使用了gcc提供的标签地址特性.这使得我们可以将`当前或包含函数中定义的`goto跳转标签视作常值数据进行操作.
>
> 可以用于线程化代码的解释器中,通过在线程代码中保存解释器标签来实现快速调度.
>
> ```c
> static void ** ptr = &&foo;
> static void * array[] = {&&foo, &&bar};
> goto array[1];
> ```
>
> **注意不能跳转到外部函数**,最好将其保存在auto变量中并不作为参数传递.
>
> ```c
> static const int array[] = { &&foo - &&foo, &&bar - &&foo,&&hack - &&foo };
> goto *(&&foo + array[i]);
> ```

#### 指令译码

```c
// macro: src[12][RI] dest[RI]
// macro: immX 取对应类型的立即数
static void decode_operand(Decode *s, word_t *dest, word_t *src1, word_t *src2, int type) {
    uint32_t i = s->isa.inst.val;
    int rd  = BITS(i, 11, 7);int rs1 = BITS(i, 19, 15);int rs2 = BITS(i, 24, 20);
    destR(rd);//设置目标位置,后续可自行覆盖
    switch(TYPE){
        case TYPE_X: ...;//使用宏设置入参为对应值
        case TYPE_S: destI(immS(i)); ...//此处为了方便将立即数放在dest中,与手册不同
    }
}
```

```c
// BITMASK(bits) 低bits位全1 ((1ull << (bits)) - 1) 
// BITS(x, hi, lo) 获取指定位[],0开始 (((x)>>(lo)) & BITMASK((hi)-(lo)+1))
#define SEXT(x, len) ({ struct { int64_t n : len; } __x = { .n = x }; (uint64_t)__x.n; }) //返回低n位的符号拓展 (return 值为最后一个语句的值)
```

> `位域`范围不能超过依赖的数据类型,同时在使用时会根据类型选择`拓展方式(有无符号)`**此处使用对有符号数类型变量进行位域操作再强制类型转换返回无符号数实现==对立即数的符号拓展==**.

### 添加指令

![image-20220721225744878](/home/summage/Course/pict/ysyx/image-20220721225744878.png)

#### 添加指令类型

![image-20220730145434263](/home/summage/Course/pict/ysyx/image-20220730145434263.png)

`src/isa/$ISA/inst.c`

```c
enum{TYPE_X,..., TYPE_N}; // 添加指令类型标志
// 添加对应译码
static void decode_operand(Decode * s, word_t *dest, word_t*src1, word_t*src2,int type){
    ...;
    switch(type){
        case TYPE_X:.....;break;
    }
}
```

```c
static word_t immX(uint32_t i);//获取对应指令类型立即数
//设置对应属性
#define src[12][RI]([ni]) do{*src[12] = R(n)/i;}while(0)
#define dest[RI]([ni]) do{*dest=R(n)/i;}while(0)
```

#### 添加指令

`src/isa/$ISA/inst.c`

```c
static int decode_exec(Decode * s){
    INSTPAT_START();
    INSTPAT("Pat 32 bits", name, type(X), operation);
```

手册上的pc是指向当前指令地址还是已经指向了下一条?

检查反汇编代码可知是当前指令位置,所以dnpc=pc+dest

### 程序，运行时环境和AM

#### 运行时环境

涉及程序内存管理、变量访问、参数传递、操作系统接口等方面，同时还一般对设置和管理堆栈(可能包括垃圾回收,线程或其他语言内置的动态功能)。编译器在生成代码时会依赖于特定的运行时系统进行假设从而生成正确的代码。

当前条件下我们可以实现的`最简单的运行时环境`便是将程序放置在正确内存位置,将pc指向程序起始位置,提供`nemu_trap`作为程序结束方式.但是这样的环境**与isa高度耦合**.

通过将直接的isa操作抽象为接口统一的API(*或称为封装为库函数*)来实现程序与架构的解耦.

#### AM裸机运行时环境

根据一套统一的API实现的ISA抽象,根据需求划分为5个模块:

* TRM图灵机:提供基本计算
* IOE输入输出拓展
* CTE上下文拓展:提供上下文管理
* VME虚存拓展:虚拟内存管理
* MPE多处理器拓展:多处理器通信

这样我们就在`nemu(硬件功能实现)`和`app(程序运行)`之间加入了`抽象层(运行时环境)`.

> AM存在的必要性(与操作系统的运行时环境的区别与原因)?
>
> AM为各种ISA提供了一套统一的抽象接口来满足程序的需求，而操作系统的运行时环境是由其特定硬件和软件组成的，即不同的操作系统其运行时环境是独立的，而AM则只需要为对应ISA实现与硬件紧密相连的部分API即可，是一个整体。从而可以将与ISA特性无关的功能单独划分出来，不用重复实现。

#### AM结构与源码

```c
#include ARCH_H // this macro is defined in $CFLAGS
                // examples: "arch/x86-qemu.h", "arch/native.h", ...
// Memory protection flags
#define MMAP_NONE  0x00000000 // no access
#define MMAP_READ  0x00000001 // can read
#define MMAP_WRITE 0x00000002 // can write
// Memory area for [@start, @end)
typedef struct {
  void *start, *end;
} Area;
// Arch-dependent processor context
typedef struct Context Context;
// An event of type @event, caused by @cause of pointer @ref
typedef struct {
  enum {
    EVENT_NULL = 0,
    EVENT_YIELD, EVENT_SYSCALL, EVENT_PAGEFAULT, EVENT_ERROR,
    EVENT_IRQ_TIMER, EVENT_IRQ_IODEV,
  } event;
  uintptr_t cause, ref;
  const char *msg;
} Event;
// A protected address space with user memory @area
// and arch-dependent @ptr
typedef struct {
  int pgsize;
  Area area;
  void *ptr;
} AddrSpace;
#ifdef __cplusplus
extern "C" {
#endif
// ----------------------- TRM: Turing Machine -----------------------
extern   Area        heap;
void     putch       (char ch);
void     halt        (int code) __attribute__((__noreturn__));

// -------------------- IOE: Input/Output Devices --------------------
bool     ioe_init    (void);
void     ioe_read    (int reg, void *buf);
void     ioe_write   (int reg, void *buf);
#include "amdev.h"

// ---------- CTE: Interrupt Handling and Context Switching ----------
bool	 (Context *(*handler)(Event ev, Context *ctx));
void     yield       (void);
bool     ienabled    (void);
void     iset        (bool enable);
Context *kcontext    (Area kstack, void (*entry)(void *), void *arg);

// ----------------------- VME: Virtual Memory -----------------------
bool     vme_init    (void *(*pgalloc)(int), void (*pgfree)(void *));
void     protect     (AddrSpace *as);
void     unprotect   (AddrSpace *as);
void     map         (AddrSpace *as, void *vaddr, void *paddr, int prot);
Context *ucontext    (AddrSpace *as, Area kstack, void *entry);

// ---------------------- MPE: Multi-Processing ----------------------
bool     mpe_init    (void (*entry)());
int      cpu_count   (void);
int      cpu_current (void);
int      atomic_xchg (int *addr, int newval);

#ifdef __cplusplus
}
#endif
#endif
```

* `abstract-machine/am/`框架的AM API实现
* `abstract-machine/klib`框架无关的库函数

#### TRM

只需要提供很少的api

* Area heap:指示堆区范围.

  > 堆区的分配和管理由程序自行完成

* void putch(char ch): 输出一个字符

* void halt(int code): 结束程序运行

  > 调用nemu_trap()这一由isa决定的宏,展开得到(riscv)`asm volatile("mv a0, %0; ebreak"::"r"(code));`.将寄存器a0中值(对应code)传递给set_nemu_state进而指示monitor程序结束原因.

* void \_trm_init(): 初始化

####　使用ＡＭ作为运行时环境

==Cross Compile交叉编译==用于在编译平台上创建将在另一种运行时环境下运行的程序的一种操作.用于机器代码的跨平台生成.

> 一般交叉编译链的命名规则为arch(目标)-core(没固定标准)-kernel(运行的OS)-system(选择的库函数和目标映像的规范,gnu=glibc+oabi,gnueabi=glibc+eabi)

`在abstract-machine下make html获得可读版本的makefile`

```makefile
WORK_DIR  = $(shell pwd)
DST_DIR   = $(WORK_DIR)/build/$(ARCH)
$(shell mkdir -p $(DST_DIR))
# 目标文件
# NAME在其他源文件相关的makefile中定义,即仅当其他makefile调用当前时才会使用IMAGE相关内容
IMAGE_REL = build/$(NAME)-$(ARCH) # ARCH=$ISA-nemu手动输入
IMAGE     = $(abspath $(IMAGE_REL))
# 链接用文件
OBJS      = # ./build/$(ARCH)/.c文件原路径.o
LIBS     := $(sort $(LIBS) am klib) # lazy evaluation ("=") causes infinite recursions # 右侧$(LIBS)为空,故结果为am klib
LINKAGE   = # $(AM_HOME)/$(LIBS)/build/$(LIBS)-$(ARCH).a
```

```makefile
# 交叉编译工具
AS        = $(CROSS_COMPILE)gcc
CC        = $(CROSS_COMPILE)gcc
CXX       = $(CROSS_COMPILE)g++
LD        = $(CROSS_COMPILE)ld
OBJDUMP   = $(CROSS_COMPILE)objdump
OBJCOPY   = $(CROSS_COMPILE)objcopy
READELF   = $(CROSS_COMPILE)readelf
# 交叉编译参数
INC_PATH += # ./include $(AM_HOME)/$(LIBS)/include/
INCFLAGS += -I $(INC_PATH) # 添加搜索路径
# -M生成文件相关的信息,包括所有依赖的源文件 -MM忽视#include类型依赖 -MMD将结果输出到%.d文件
# 
CFLAGS += -O2 -MMD -Wall -Werror $(INCFLAGS) \
            -D__ISA__=\"$(ISA)\" -D__ISA_$(shell echo $(ISA) | tr a-z A-Z)__ \
            -D__ARCH__=$(ARCH) -D__ARCH_$(shell echo $(ARCH) | tr a-z A-Z | tr - _) \
            -D__PLATFORM__=$(PLATFORM) -D__PLATFORM_$(shell echo $(PLATFORM) | tr a-z A-Z | tr - _) \
            -DARCH_H=\"arch/$(ARCH).h\" \
            -fno-asynchronous-unwind-tables -fno-builtin -fno-stack-protector \
            -Wno-main -U_FORTIFY_SOURCE
CXXFLAGS +=  $(CFLAGS) -ffreestanding -fno-rtti -fno-exceptions
ASFLAGS  += -MMD $(INCFLAGS)
```

```makefile
# 框架相关代码
-include $(AM_HOME)/scripts/$(ARCH).mk
# 无交叉编译器则退回普通编译器
ifeq ($(wildcard $(shell which $(CC))),)
  $(info #  $(CC) not found; fall back to default gcc and binutils)
  CROSS_COMPILE :=
endif
```



```makefile
# 编译顺序控制
image: image-dep
archive: $(ARCHIVE)
image-dep: $(OBJS) am $(LIBS)
	@echo \# Creating image [$(ARCH)]
.PHONY: image image-dep archive run $(LIBS)
```



##### 编译

**调用路径**：

1. `nemu`下makefile配置编译选项,而编译目标均在`./scripts/%.mk`中,通过调用该目录下的次级mk完成编译.

   > `native.mk`层级最高,直接面向用户的编译目标.
   >
   > `build.mk`提供从源文件到目标文件再经由链接得到可执行文件的规则.
   >
   > `config.mk`则用于配置选项以及生成响应文件

2. `abstract-machine`为不同isa和平台提供基本编译规则但是不包含运行程序的源文件,需要被其他makefile调用,作为中间层存在(负责编译依赖,以及交叉编译生成).

   > `./Makefile`提供了目标文件生成,依赖库归档生成,并进一步生成elf文件与可执行文件的规则.**但是这些规则中的许多目标参数都是由调用他的上层mk或被他调用的次级mk提供.**
   >
   > `./xxlib/Makefile`提供给上层Makefile`NAME`和`SRCS`参数.一般由`./Makefile`通过-C参数指定来调用,而自己又include`./Makefile`来完成递归调用完成依赖库的打包
   >
   > `./scripts/%.mk`为isa-平台相关内容的mk,由命令行参数指定调用其中一个,提供`附加SCRS和编译选项-DISA_H=`,同时include次级目录下的isa和平台mk.**平台mk**基本就是上面的nemu makefile,而**isa mk**则提供了**交叉编译工具链部分名称**,编译和链接参数.

3. `am-kernels/tests/cpu-test`根据指定的文件生成对应Makefile(指定NAME,SCRS并include$(AM_HOME)/Makefile).通过调用abtract-machine作为中间层让其通过nemu运行.即实际的编译规则都在`abstract-machine/Makefile`中,此处设置对应规则调用即可.

将`$ISA-nemu`的AM实现源文件编译为目标文件

```makefile
# 单个文件的编译规则c(CC) cc(CXX) cpp(CXX) S(AS)类似,只有-std=内容以及echo内容的差别
$(DST_DIR)/%.o: %.c
	@mkdir -p $(dir $@) && echo + CC $<
	@$(CC) -std=gnu11 $(CFLAGS) -c -o $@ $(realpath $<)
```

使用ar将AM目标文件作为库打包成一个归档文件(**静态链接库**)

```makefile
$(ARCHIVE): $(OBJS) # $(ARCHIVE) am-$ISA-nemu.a
	@echo + AR "->" $(shell realpath $@ --relative-to .)
	@ar rcs $(ARCHIVE) $(OBJS)  # $OBJ为编译生成的.o文件
```

> `ar`用于集合文件生成备份\归档文件,同时保有原来的属性和权限
>
> r 插入指定文件到备份中(**替换**) 
>
> c 建立备份文件.一般在请求更新(an update)时,不存在的存档文件总会被创建同时产生warning.而创建时使用这一参数可以关闭这一warning.
>
> s 若包含对象模式则建立符号表(S则不产生符号表)



将`应用程序`源文件编译目标文件

将`依赖的运行库`(如/abstract-machine/klib)编译并打包成归档文件

```makefile
# 递归编译依赖库
$(LIBS): %:
	@$(MAKE) -s -C $(AM_HOME)/$* archive
$(ARCHIVE): $(OBJS)
	@echo + AR "->" $(shell realpath $@ --relative-to .)
	@ar rcs $(ARCHIVE) $(OBJS)
archive: $(ARCHIVE)
# $(AM_HOME)/klib/Makefile $(AM_HOME)/am/Makefile
NAME = klib
SRCS = $(shell find src/ -name "*.c")
include $(AM_HOME)/Makefile
# 此处通过klib,am调用$(ARCHIVE)打包,则此时SCRS已经替换为了klib,am的源文件,根据依赖链条,会由上层makefile完成编译与打包
```



根据`abstract-machine/scripts/$ISA-nemu.mk`,让ld根据连接脚本`linker.ld`将目标文件和归档文件链接成可执行文件.

> 在链接脚本中`_pmem_start和_entry_offset`决定了可执行程序重定位后的节的起始位置(0x100000或0x80000000).

```makefile
# 将目标文件和静态依赖库打包为elf文件,并后续连接生成image
$(IMAGE).elf: $(OBJS) am $(LIBS)
	@echo + LD "->" $(IMAGE_REL).elf
	@$(LD) $(LDFLAGS) -o $(IMAGE).elf --start-group $(LINKAGE) --end-group
```

```makefile
# abstract-machine/scripts/native.mk # 运行时环境相关makefile
CFLAGS  += -fpie
ASFLAGS += -fpie -pie
# -l链接静态库
# -Wl间接调用linker(ld)来链接动态库 -Wl,--startgrop xxx.o .. -wl,--endgroup(或下面那个形式)
# whole-archive将之后提到的库里的目标文件全部链接进来而不是搜索需要的目标文件 no-whole-archive关闭上述设置,但是gcc并不知道这一规则,所以需要使用以下格式调用
# -Wl,--whole-archive linkage -Wl,-no-whole-archive
image:
	@echo + LD "->" $(IMAGE_REL)
	# 	pie:Produce a position independent executable on targets which support it.
	@g++ -pie -o $(IMAGE) -Wl,--whole-archive $(LINKAGE) -Wl,-no-whole-archive -lSDL2 -ldl
run: image
	$(IMAGE)
gdb: image
	gdb -ex "handle SIGUSR1 SIGUSR2 SIGSEGV noprint nostop"
```

##### 添加批模式运行

`nemu/scripts/native.mk`添加batch选项，在shell中调用时附加-b即可

对于`am-kernels`，首先在当前目录的makefile中加入该`batch`模式。由于其实际编译规则存在于`abstract-machine`中,所以需要在am中加入对应规则.

在`am`中,与nemu相关的编译选项存在于`scripts/platform/nemu.mk`中,需要加入对应选项并修改ARGS附加-b选项.

##### 链接脚本

```assembly
# abstract-machine/scripts/linker.ld
# this declares that the symbol from boot.S is the entry point to our code. `_start`
ENTRY(_start)  

SECTIONS {
  /* _pmem_start and _entry_offset are defined in LDFLAGS */
  . = _pmem_start + _entry_offset; # 用pmemstart和entry offset确定程序存放的起始位置
  .text : { # 代码段
    *(entry) # 以abstract-machine/am/$ISA/src/nemu/start.S中定义的entry为起始，保证程序能从start.S正确开始
    *(.text*)
  }
  etext = .; # 表示将这个分配到当前地址
  _etext = .;
  .rodata : { # 只读数据,存放全局常量
    *(.rodata*)
  }
  .data : { # 在编译时初始化的全局变量的存放位置
    *(.data)
  }
  edata = .;
  _data = .;
  .bss : { # 未初始化的全局变量的位置
	_bss_start = .;
    *(.bss*)
    *(.sbss*)
    *(.scommon)
  }
  _stack_top = ALIGN(0x1000);
  . = _stack_top + 0x8000;
  _stack_pointer = .;
  end = .;
  _end = .;
  _heap_start = ALIGN(0x1000);
}
```

```assembly
.section entry, "ax"
.globl _start
.type _start, @function

_start: # 设置栈顶指针,并跳转到_trm_init函数
  mv s0, zero
  la sp, _stack_pointer
  jal _trm_init # 调用main并将结果保存在ret用于指示halt
```



##### 实现常用库函数

### 基础设施

#### trace

```c
// include/utils.h
#define log_write(...) IFDEF(CONFIG_TARGET_NATIVE_ELF, \
  do { \
    extern FILE* log_fp; \
    extern bool log_enable(); \
    if (log_enable()) { \
      fprintf(log_fp, __VA_ARGS__); \
      fflush(log_fp); \
    } \
  } while (0) \
)
// sr
bool log_enable() {
  return MUXDEF(CONFIG_TRACE, (g_nr_guest_inst >= CONFIG_TRACE_START) &&
         (g_nr_guest_inst <= CONFIG_TRACE_END), false);
}
```

**IRITRACE MTRACE**比较简单

由于语句会被拆分和优化,**函数**才能较为清晰地携带程序语义.对于==FTRACE==,我们可以通过识别对应PC值并将其翻译为函数名从而得到调用信息.而保存在**ELF文件中的符号表(symbol table)**则记录了编译时的信息,包括变量,函数等信息.

`riscv64-linux-gnu-readelf -a add-riscv32-nemu.elf`

```
Symbol table '.symtab' contains 28 entries:
   Num:    Value          Size Type    Bind   Vis      Ndx Name
     0: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT  UND 
     1: 0000000080000000     0 SECTION LOCAL  DEFAULT    1 .text
     2: 0000000080000128     0 SECTION LOCAL  DEFAULT    2 .srodata.mainargs
     3: 0000000080000130     0 SECTION LOCAL  DEFAULT    3 .data.ans
     4: 0000000080000230     0 SECTION LOCAL  DEFAULT    4 .data.test_data
     5: 0000000000000000     0 SECTION LOCAL  DEFAULT    5 .comment
     6: 0000000000000000     0 FILE    LOCAL  DEFAULT  ABS add.c
     7: 0000000000000000     0 FILE    LOCAL  DEFAULT  ABS trm.c
     8: 0000000080000128     1 OBJECT  LOCAL  DEFAULT    2 mainargs
     9: 0000000080000108    32 FUNC    GLOBAL DEFAULT    1 _trm_init
    10: 0000000080009000     0 NOTYPE  GLOBAL DEFAULT    4 _stack_pointer
    11: 0000000080000128     0 NOTYPE  GLOBAL DEFAULT    1 _etext
    12: 0000000080000000     0 NOTYPE  GLOBAL DEFAULT  ABS _pmem_start
    13: 0000000080000250     0 NOTYPE  GLOBAL DEFAULT    4 _bss_start
    14: 0000000080000129     0 NOTYPE  GLOBAL DEFAULT    2 edata
    15: 0000000080009000     0 NOTYPE  GLOBAL DEFAULT    4 _heap_start
    16: 0000000080001000     0 NOTYPE  GLOBAL DEFAULT    4 _stack_top
    17: 0000000080009000     0 NOTYPE  GLOBAL DEFAULT    4 end
.....................
```

> **什么是符号？**
>
> 符号代表函数或变量的地址，用于在链接过程中的重定位。从作用可知，存在于符号表中的变量必须是在编译链接后有固定地址的变量。所以全局变量和静态变量会出现在符号表中，而运行时分配在堆栈中的局部变量则不会出现。

> **符号表的作用阶段**:
>
> 从目标文件开始就出现符号表,但此时的符号表value均为0,即尚未指定函数和变量的地址.而链接过程会根据所有目标和库文件的符号表分配各个表项的位置,进而生成完整的分配完成的符号表,并替换预先填入的符号.此时符号表更多地是方便查阅,对运行已经没有影响.

> **ELF文件格式**
>
> 

> **部分函数调用没有对应ret**
>
> 对应尾部固定跳转另外函数的函数，由于其使用j进行函数跳转，所以只需要一个ret跳出最外层。

> **为什么定义了__NATIVE_USE_KLIB__之后就会链接到klib?**
>
> 首先需要补充链接时**强/弱符号/引用的概念**.
>
> 对于全局变量,默认有初始化的为强符号而没有初始化的为弱符号.
>
> 对于函数,需要使用\_\_attribute\_\_((weak))定义弱符号,否则默认为强符号.而\_\_attribute\_\_((weakref("替代函数名"))).
>
> 1) 强符号不可重复定义否则链接器报错
> 2) 在强弱符号同时存在时,编译器会首选强符号.
> 3) 有且只有多个弱符号时选择占用空间最大的一个
>
> 而弱连接在指定的替代函数未被实现时是未定义的**(if(函数名)判定为否)**,而在替代函数实现时等同于替代函数.**即可以暂时搁置实现,统一指定一个信息打印或报错函数,等有时间再使用强引用函数重载**.
>
> **标准库中的实现一般都是弱的,从而方便用户重载**
>
> 定义了\_\_NATIVE\_USE\_KLIB\_\_后自己的函数定义才存在(否则在预处理阶段被裁掉了),作为强符号(引用?)重载了标准库函数.
>
> > `-fno-builtin`编译选项与链接
> >
> > gcc实现了部分更高效的内置函数用于替换部分库函数，这些内置函数是最强的，所以想要替换此类函数时需要使用该选项否则一律拐到内置函数



#### DiffTest

实现`isa_difftest_checkregs`即可.由于Spike顺序和riscv64一致,直接比对即可.注意pc为执行之前的pc,与ref比对需要获取当前pc.

### 输入输出

> volatile

在`ab.../am/src/platform/nemu/include/nemu.h`中定义了各个设备端口的地址.

#### 时钟

> **real-time test**
>
> 查看am-tests下src/main.c可知,提供mainargs=t即可运行
>
> ```c
> CASE('i', hello_intr, IOE, CTE(simple_trap));
> //展开
> case 'i': {
>  void hello_intr();
>  entry = hello_intr;
>  ({ ioe_init(); }), ({
>    Context *simple_trap(Event, Context *);
>    cte_init(simple_trap);
>  });
>  hello_intr();
>  break;
> };
> enum { AM_TIMER_CONFIG = (4) };
> typedef struct {
> _Bool present, has_rtc;
> } AM_TIMER_CONFIG_T;
> ;
> enum { AM_TIMER_RTC = (5) };
> typedef struct {
> int year, month, day, hour, minute, second;
> } AM_TIMER_RTC_T;
> ;
> enum { AM_TIMER_UPTIME = (6) };
> typedef struct {
> uint64_t us;
> } AM_TIMER_UPTIME_T;
> ```
>
> 

阅读`nemu/src/device/timer.c`得知,`init_timer`函数会根据是否有io端口来映射并初始化`rtc_port_base`(直接分配的,地址与端口映射地址无关,读写端口映射地址会经过函数传递(`fetch_mmio_map等`)改为读写该地址), 将`rtc_io_handler`函数设置为回调函数.该函数在特定输入下会调用`get_time`函数获得当前运行的毫秒数(64位)并保存在`rtc_port_base[1] | rtc_port_base[0]`(小字节序)中.

那么要获取其保存的运行时间信息,我们需要从两个端口位置读取数据.对于`native`使用的`io_read`实际是在调用`__am_timer_uptime`函数.

观察回调函数可知,当使用inl每次读取32位信息时,仅在读取高位时会触发时钟记录更新,所以读取时要先读高32位.`高位更新可以将读取顺序错误时的最大误差限制在低32位`

```c
static void rtc_io_handler(uint32_t offset, int len, bool is_write) {
  assert(offset == 0 || offset == 4);
  if (!is_write && offset == 4) {
    uint64_t us = get_time();
    rtc_port_base[0] = (uint32_t)us;
    rtc_port_base[1] = us >> 32;
  }
}
```

<img src="/home/summage/Course/pict/ysyx/CoreMark跑分_native.png" alt="CoreMark跑分_native" style="zoom:50%;" /><img src="/home/summage/Course/pict/ysyx/CoreMark跑分_nemu.png" alt="CoreMark跑分_nemu" style="zoom: 50%;" />

<img src="/home/summage/Course/pict/ysyx/Dhrystone跑分_native.png" alt="Dhrystone跑分_native" style="zoom:50%;" /><img src="/home/summage/Course/pict/ysyx/Dhrystone跑分_nemu.png" alt="Dhrystone跑分_nemu" style="zoom:50%;" />

<img src="/home/summage/Course/pict/ysyx/MicroBench跑分_native.png" alt="MicroBench跑分_native" style="zoom:50%;" /><img src="/home/summage/Course/pict/ysyx/MicroBench_nemu跑分.png" alt="MicroBench_nemu跑分" style="zoom:50%;" />

#### 键盘输入

```c
// ab../am/include/amdev.h
#define AM_KEYS(_) \
	_(1) _(2) ...
#define AM_KEY_NAMES(key) AM_KEY_##key
enum{
    AM_KEY_HOME=0,
    AM_KEYS(AM_KEY_NAMES)
}

typedef struct {
  _Bool keydown;
  int keycode;
} AM_INPUT_KEYBRD_T;
```

AM_KEY_NAMES宏将传入的参数key(本身视作字符串)加上前缀形成AM_KEY_key，而AM_KEYS宏则会调用传入的函数并提供指定好的参数。define在存在多个调用行为时，会自动补充`,`隔开。

```c
void __am_input_keybrd(AM_INPUT_KEYBRD_T *kbd) {
  kbd->keycode = inl(KBD_ADDR);
  kbd->keydown = (kbd->keycode & KEYDOWN_MASK ? true : false);
  kbd->keycode &= ~KEYDOWN_MASK;
}
```

考虑到键盘最多108个键位，可以使用两个uint64_t来做标志。在输入为AM_KEY_NONE或为长按键位时正常按顺序检查输出已经按下的键位，从而实现多个键同时识别的效果。

####　VGA 

`nemu/src/device/vga.c`中定义了相关API与寄存器.

```c
void init_vga() {
  vgactl_port_base = (uint32_t *)new_space(8); // uint32_t VGA_CTL端口共长8个字节,即两个元素
  vgactl_port_base[0] = (screen_width() << 16) | screen_height();//低16位为高,高16位为宽
#ifdef CONFIG_HAS_PORT_IO
  add_pio_map ("vgactl", CONFIG_VGA_CTL_PORT, vgactl_port_base, 8, NULL);
#else
  add_mmio_map("vgactl", CONFIG_VGA_CTL_MMIO, vgactl_port_base, 8, NULL); 
#endif

  vmem = new_space(screen_size());
  add_mmio_map("vmem", CONFIG_FB_ADDR, vmem, screen_size(), NULL);
  IFDEF(CONFIG_VGA_SHOW_SCREEN, init_screen());
  IFDEF(CONFIG_VGA_SHOW_SCREEN, memset(vmem, 0, screen_size()));
}
//在ctl->sync==false时各个属性：x y 为绘制区域左上角坐标，w h为绘制区域大小，pixels为需要绘制的像素(行优先存储)
//ctl->sync==true时其他均为0
fb[(row+y)*width+x+col] = pixels[row*w+col];
```

```c
enum { AM_UART_CONFIG = (1) };
typedef struct {
  _Bool present;
} AM_UART_CONFIG_T;
;
enum { AM_UART_TX = (2) };
typedef struct {
  char data;
} AM_UART_TX_T;
;
enum { AM_UART_RX = (3) };
typedef struct {
  char data;
} AM_UART_RX_T;
;
enum { AM_TIMER_CONFIG = (4) };
typedef struct {
  _Bool present, has_rtc;
} AM_TIMER_CONFIG_T;
;
enum { AM_TIMER_RTC = (5) };
typedef struct {
  int year, month, day, hour, minute, second;
} AM_TIMER_RTC_T;
;
enum { AM_TIMER_UPTIME = (6) };
typedef struct {
  uint64_t us;
} AM_TIMER_UPTIME_T;
;
enum { AM_INPUT_CONFIG = (7) };
typedef struct {
  _Bool present;
} AM_INPUT_CONFIG_T;
;
enum { AM_INPUT_KEYBRD = (8) };
typedef struct {
  _Bool keydown;
  int keycode;
} AM_INPUT_KEYBRD_T;
;
enum { AM_GPU_CONFIG = (9) };
typedef struct {
  _Bool present, has_accel;
  int width, height, vmemsz;
} AM_GPU_CONFIG_T;
;
enum { AM_GPU_STATUS = (10) };
typedef struct {
  _Bool ready;
} AM_GPU_STATUS_T;
;
enum { AM_GPU_FBDRAW = (11) };
typedef struct {
  int x, y;
  void *pixels;
  int w, h;
  _Bool sync;
} AM_GPU_FBDRAW_T;
;
enum { AM_GPU_MEMCPY = (12) };
typedef struct {
  uint32_t dest;
  void *src;
  int size;
} AM_GPU_MEMCPY_T;
;
enum { AM_GPU_RENDER = (13) };
typedef struct {
  uint32_t root;
} AM_GPU_RENDER_T;
;
enum { AM_AUDIO_CONFIG = (14) };
typedef struct {
  _Bool present;
  int bufsize;
} AM_AUDIO_CONFIG_T;
;
enum { AM_AUDIO_CTRL = (15) };
typedef struct {
  int freq, channels, samples;
} AM_AUDIO_CTRL_T;
;
enum { AM_AUDIO_STATUS = (16) };
typedef struct {
  int count;
} AM_AUDIO_STATUS_T;
;
enum { AM_AUDIO_PLAY = (17) };
typedef struct {
  Area buf;
} AM_AUDIO_PLAY_T;
;
enum { AM_DISK_CONFIG = (18) };
typedef struct {
  _Bool present;
  int blksz, blkcnt;
} AM_DISK_CONFIG_T;
;
enum { AM_DISK_STATUS = (19) };
typedef struct {
  _Bool ready;
} AM_DISK_STATUS_T;
;
enum { AM_DISK_BLKIO = (20) };
typedef struct {
  _Bool write;
  void *buf;
  int blkno, blkcnt;
} AM_DISK_BLKIO_T;
;
enum { AM_NET_CONFIG = (21) };
typedef struct {
  _Bool present;
} AM_NET_CONFIG_T;
;
enum { AM_NET_STATUS = (22) };
typedef struct {
  int rx_len, tx_len;
} AM_NET_STATUS_T;
;
enum { AM_NET_TX = (23) };
typedef struct {
  Area buf;
} AM_NET_TX_T;
;
enum { AM_NET_RX = (24) };
typedef struct {
  Area buf;
} AM_NET_RX_T;
;
```

#### 声卡

1. `SDL_OpenAudio()`根据提供的频率,格式等参数初始化音频系统,同时注册提供音频数据的回调函数.[SDL](https://wiki.libsdl.org/SDL_OpenAudio)
2. `SDL库`提供缓冲区并定期调用回调函数,请求数据写入
3. `SDL库`根据初始参数播放缓冲区内数据

```c
// the buffer is not promised to be inited
// once returned, the buffer expired.
// stereo samples are stored in a LRLR ordering
void MyAudioCallback(void * userdata,// parameters
                     uint8_t * stream, // buffer
                     int len); // buffer size

SDL_AudioSpec want, have;

SDL_memset(&want, 0, sizeof(want)); /* or SDL_zero(want) */
want.freq = 48000; // the num of sampled frames per sec
want.format = AUDIO_F32;
want.channels = 2;
want.samples = 4096; // a unit of audio data or the buffer size when used with OpenAudioDevice
want.callback = MyAudioCallback; 

if (SDL_OpenAudio(&want, &have) < 0) {
    SDL_Log("Failed to open audio: %s", SDL_GetError());
} else {
    if (have.format != want.format) {
        SDL_Log("We didn't get Float32 audio format.");
    }
    SDL_PauseAudio(0); /* start audio playing. */
    SDL_Delay(5000); /* let the audio callback play some sound for 5 seconds. */
    SDL_CloseAudio();
}
int SDL_OpenAudio(SDL_AudioSpec * desired, SDL_AudioSpec * obtained);

```

> 通过io从程序中获取参数`nemu/src/device/audio.c`
>
> * freq,channels,samples提供参数
> * init在被写入后会根据寄存器中参数初始化
> * STREAM_BUF存放程序提供的音频数据
> * sbuf_size,count分别为缓冲区大小和已使用大小
>
> nemu在初始化声卡时会通过`new_space`函数初始化24字节的`audio_space`端口(6个)空间(0x200或0xa1200000).并在MMIO中注册64kb的缓冲区.("audio","audio-sbuf")
>
> ```c
> static uint8_t *io_space = NULL;//io空间起始地址
> static uint8_t *p_space = NULL;//当前空闲位置
> //在空间充足的情况下为端口分配[p_space,p_space+size]段
> uint8_t* new_space(int size) {
>   uint8_t *p = p_space;
>   // page aligned;
>   size = (size + (PAGE_SIZE - 1)) & ~PAGE_MASK;
>   p_space += size;
>   assert(p_space - io_space < IO_SPACE_MAX);
>   return p;
> }
> ```

> `abstract-machine端口`
>
> * AM_AUDIO_CONFIG 声卡控制器信息:present,bufsize(假设运行中不变)
> * AM_AUDIO_CTRL 控制寄存器,根据写入的freq,channels,samples初始化
> * AM_AUDIO_STATUS 状态,读count
> * AM_AUDIO_PLAY 在流缓冲区剩余空间充足时将将buf中数据写入



### typing game

`chars`结构体包含字符以及当前坐标,速度,方向.

帧数由FPS宏控制,而CPS宏则决定了刷新时间间隔(当%(FPS/CPS)时触发刷新).

`game_logic_update(frame)`

> 遍历chars数组对非空元素更新状态.对未到达的,按照速度更新高度.
>
> 若滞留时间为0(未触底),超出上边的设置为空,而触底的将速度置零,高度规范,时间设置为FPS(停留1秒后清除).
>
> 滞留时间不为0则只减少时间,并在归零后设置为空.
>
> 接收当前帧数判断是否需要刷新.刷新时调用`new_char()`遍历chars数组并填充为使用的元素,x以及速度v随机生成(`v = (screen_h - CHAR_H + 1) / randint(FPS * SPEED_FAST, FPS * SPEED_SLOW)`).

`check_hit`

> 通过遍历判断输入的字符是否与正在下落的字符匹配,是则命中,并将对应chars给予一个足够大的反速度.否则错误.

`render`

> 根据字符的类型,位置和速度方向从预先设置好的texture数组中取得数据并写入显存中.

## PA3

PA中将上下文管理抽象为CTE(context extension).

1. 执行流切换原因

   ```c
   typedef struct Evnet{
       enum{...} event; // 事件类型
       uintptr_t cause, ref;
       const char * msg;
   }
   ```

   

2. 上下文内容

   ```c
   //abstract-machine/src/include/riscv64-nemu
   //架构相关,通过CTE的API访问,尽量避免直接引用成员(破坏可移植性)
   //寄存器顺序按照trap.S排列
   struct Context {
     uintptr_t gpr[32], mcause, mstatus, mepc;
     void *pdir;
   };
   #define GPR1 gpr[17] // a7
   #define GPR2 gpr[0]
   #define GPR3 gpr[0]
   #define GPR4 gpr[0]
   #define GPRx gpr[0]
   
   // trap.S
   #   MAP(REGS, PUSH)
   
   #   csrr t0, mcause
   #   csrr t1, mstatus
   #   csrr t2, mepc
   
   ```

   

定义`HAS_CTE`后nanos-lite会调用`init_irq`函数.而该函数会调用am提供的`cte_init`,同时将异常处理函数`do_event`的地址作为参数传递.由`bool cte_init(Context*(*handler)(Event, Context*)) `来设置异常入口地址为`__am_asm_trap`(由trap.S声明并定义),并将传入的参数注册为`event handler(事件处理回调函数)`.

### 实现自陷操作

```c
// nemu/src/isa/riscv64/include/isa_def.h
typedef struct{
    word_t gpr[32], mcause, mstatus, mepc;
    vaddr_t pc;
    void * pdir;
} riscv64_CPU_state;
// nemu/src/cpu/cpu_exec.c
CPU_state cpu = {.mstatus = 0xa000018000};// 用于支持difftest
// nemu/src/isa/riscv64/local_include/reg.h
#define MCAUSE (0b001101000010)
#define mcause_ptr (&cpu.mcause)
...
#define mtvec_ptr (cpu.pdir)
```

需要添加csrr系列的指令用于访问处理控制寄存器，由于此处只有上述几种，直接根据csr的值处理对应寄存器即可，无需索引。

> AM栈大小
>
> todo

> 异常号保存可以使用软件保存吗
>
> todo

> 对比异常处理和函数调用需要保存的寄存器的原因
>
> todo

> \_\_am_irq\_handle()中使用的上下文指针以及这个指针指向的成员是在哪里赋值的？
>
> 该函数由\_\_am_asm\_trap函数调用，该函数会先在栈上分配context结构体所需空间并按照一定顺序将寄存器压入栈顶并将起始位置作为指针地址赋给a0.
>
> 在这之中，trap.S定义了保存规则并返回了该部分内容起始处的指针.riscv64-nemu.h定义了如何解析这段内存,所以结构体中元素排列顺序要与trap.S中保存一致.

> +4应该交给硬件还是软件来做

### 用户程序

`navy-apps/libs/libos/src/crt0/start/$ISA.S`中定义了起始位置,并在将s0置0后跳转至call_main函数.

链接用户程序时需要注意避免与Nanos-lite本身内容发生冲突.(此处定义从0x83000000开始).

> 在navy-apps/tests/下对应文件中使用`make ISA=$ISA`生成的可执行文件复制到nanos-lite/build下并改名为randisk.img.`nanos-lite/src/resources.S`会将此映像文件包括进来.
>
> ```assembly
> .section .data
> .global ramdisk_start, ramdisk_end
> ramdisk_start:
> .incbin "build/ramdisk.img"
> ramdisk_end:
> ```
>
> 
>
> 对于dummy来说,此镜像文件只有一个文件,即dummy程序,位于偏移0处.

> 堆栈的位置?

> 识别不同格式的可执行文件?
>
> 可执行文件头几个字节为对应的魔数.

ELF文件提供了两个视角来组织一个可执行文件, 一个是面向链接过程的section视角, 这个视角提供了用于链接与重定位的信息(例如符号表); 另一个是面向执行的segment视角, 这个视角提供了用于加载可执行文件的信息.  一个segment可能由0个或多个section组成, 但一个section可能不被包含于任何segment中.

ELF中采用program header table来管理segment, program header table的一个表项描述了一个segment的所有属性, 包括类型, 虚拟地址, 标志, 对齐方式, 以及文件内偏移量和segment大小. 根据这些信息, 我们就可以知道需要加载可执行文件的哪些字节了, 同时我们也可以看到, 加载一个可执行文件并不是加载它所包含的所有内容, 只要加载那些与运行时刻相关的内容就可以了, 例如调试信息和符号表就不必加载. 我们可以通过判断segment的`Type属性`是否为`PT_LOAD`来判断一个segment是否需要加载.

> segment的`filesize`和`memsize`的差异(后者一般不小于前者)多是由程序数据段造成的.
>
> 由于数据段中包含已初始化和未初始化的程序变量,但只有已初始化的值才会出现在可执行文件中,而未初始化变量被保存在`.bss`的零大小部分.而装载到内存后二者都需要占用空间.

```
      +-------+---------------+-----------------------+
      |       |...............|                       |
      |       |...............|                       |  ELF file
      |       |...............|                       |
      +-------+---------------+-----------------------+
      0       ^               |              
              |<------+------>|       
              |       |       |             
              |       |                            
              |       +----------------------------+       
              |                                    |       
   Type       |   Offset    VirtAddr    PhysAddr   |FileSiz  MemSiz   Flg  Align
   LOAD       +-- 0x001000  0x03000000  0x03000000 +0x1d600  0x27240  RWE  0x1000
                               |                       |       |     
                               |   +-------------------+       |     
                               |   |                           |     
                               |   |     |           |         |       
                               |   |     |           |         |      
                               |   |     +-----------+ ---     |     
                               |   |     |00000000000|  ^      |   
                               |   | --- |00000000000|  |      |    
                               |   |  ^  |...........|  |      |  
                               |   |  |  |...........|  +------+
                               |   +--+  |...........|  |      
                               |      |  |...........|  |     
                               |      v  |...........|  v    
                               +-------> +-----------+ ---  
                                         |           |     
                                         |           |    
                                            Memory  
```



这个segment使用的内存就是`[VirtAddr, VirtAddr + MemSiz)`这一连续区间, 然后将segment的内容从ELF文件中读入到这一内存区间, 并将`[VirtAddr + FileSiz, VirtAddr + MemSiz)`对应的物理区间清零.

> 清零操作对应将未初始化变量默认初始化为0

### 实现系统调用

`abstrace-machine/am/src/riscv/nemu/cte.c/__am_irq_handle`在已注册用户处理程序(do_event)的情况下根据c->mcause识别异常类型并在分发后传递给用户处理回调函数.并返回`上下文地址`

`nanos-lite/src/irq.c/do_event`根据异常类型选择直接处理或分发给`do_syscall`做进一步处理.返回`上下文地址`.

`nanos-lite/src/syscall.c/do_syscall`则根据`a7`寄存器的值识别不同的系统调用(定义在同目录下的`syscall.h`中),以`a1,a2,a3`为参数,将结果存放在`a0`中.

而运行的程序由`navy-apps`编译而来,其中需要设置对syscall的调用接口.`navy-apps/libs/libos/src/syscall.c`

> 在上述文件内也可以找到GPR系列对应的寄存器
>
> `# define ARGS_ARRAY ("ecall", "a7", "a0", "a1", "a2", "a0")`

### 文件系统

> 修改`nanos-lite/Makefile`，HAS_NAVY = 1
>
> 然后运行`make ARCH=$ISA-nemu update`来编译navy中的程序，并把`navy-apps/fsimg/`目录下的所有内容整合成ramdisk镜像navy-apps/build/ramdisk.img, 同时生成这个ramdisk镜像的文件记录表`navy-apps/build/ramdisk.h`, Nanos-lite的Makefile会通过软连接把它们链接到项目中.

```c
// 文件记录表
typedef struct{
    char * name;
    size_t size;
    size_t disk_offset;
    size_t open_offset;
} Finfo;
// API
int open(const char * pathname, int flags, int mode);
size_t read(int fd, const void *buf, size_t len);
int close(int fd);
size_t lseek(int fd, size_t offset, int whence);
#define FD_STDxx 0-2

int fs_open(const char *pathname, int flags, int mode);//未找到则触发异常 
size_t fs_read(int fd, void *buf, size_t len);//允许所有读
size_t fs_write(int fd, const void *buf, size_t len);//允许所有写
size_t fs_lseek(int fd, size_t offset, int whence);
int fs_close(int fd);//未维护状态,直接返回0
```

由于文件数量是固定的,所以直接返回下标作为fd即可.为每个文件添加`open_offset`方便多次读写.

> 将open_offset置于文件维护表中会导致部分功能无法正常运作?

> 选择open还是fopen?
>
> ```c
> FILE *_fopen_r (...){
> 	......
> fp->_file = f;
> fp->_flags = flags;
> fp->_cookie = (void *) fp;
> fp->_read = __sread;
> fp->_write = __swrite;
> fp->_seek = __sseek;
> fp->_close = __sclose;
> 	........
> return fp;
> }
> FILE *fopen (const char *file,const char *mode){
> return _fopen_r (_REENT, file, mode);
> }
> ```
>
> 上述代码中，f对应fs_open返回的fd句柄。

### 运行时环境拓展

> 在navy-apps中通过 make ISA=native run可以在linux native上运行,从而检查自己实现的除libos与libc(Newlib)的lib实现是否正确,屏蔽了nemu硬件模拟中可能的bug的干扰.

#### 浮点数表示(fixedpt)

```
31  30                           8          0
+----+---------------------------+----------+
|sign|          integer          | fraction |
+----+---------------------------+----------+ int32_t
负实数取补码来表示，通过float*2^8来近似模拟浮点数
```

> fixedpt和float
>
> fixedpt的模拟为了简化处理,舍弃了表达范围绝对值的上下限,带来了精度损失,在整数部分较小时尤为严重.

同类型加减法可直接进行,但乘除法需要除乘一个FIXEDPT_ONE

```c
#if FIXEDPT_BITS == 32
typedef int32_t fixedpt;
typedef	int64_t	fixedptd;
typedef	uint32_t fixedptu;
typedef	uint64_t fixedptud;
#endif
#define FIXEDPT_ONE	((fixedpt)((fixedpt)1 << FIXEDPT_FBITS)) 
#define fixedpt_rconst(R) ((fixedpt)((R) * FIXEDPT_ONE + ((R) >= 0 ? 0.5 : -0.5)))
```

> *fixedpt_rconst编译后没有浮点数运算指令?*
>
> 由于fixedpt_rconst的最终结果为int类型(强转),所以编译器做了相关优化
>
> *如何实现fixedpt fixedpt_fromfloat(void \*p), 其中该指针指向一个32位浮点数,且在表达范围内*
>
> 将p强转为对应int32类型,通过位操作识别小数点位置并通过移位得到fixedpt

### 更多应用

#### NSlider

```c
typedef struct {
	int16_t x, y;
	uint16_t w, h;
} SDL_Rect;

typedef struct {
	uint32_t flags;
	SDL_PixelFormat *format;
	int w, h;
	uint16_t pitch;
	uint8_t *pixels;
} SDL_Surface;

typedef struct {
	int ncolors;
	SDL_Color *colors;
} SDL_Palette;

typedef union {
  struct {
    uint8_t r, g, b, a;
  };
  uint32_t val;
} SDL_Color;

typedef struct {
	SDL_Palette *palette;
	uint8_t BitsPerPixel;
	uint8_t BytesPerPixel;
	uint8_t Rloss, Gloss, Bloss, Aloss;
	uint8_t Rshift, Gshift, Bshift, Ashift;
	uint32_t Rmask, Gmask, Bmask, Amask;
} SDL_PixelFormat;
```



#### NTerm

main函数完成SDL、字体、terminal的初始化，并根据参数数量选择内部或外部sh_run。

内部首先输出信息以sh>标志。之后不断检测是否有按键事件触发，是则调用main中handle_key函数的ev重载版本。首先处理shift按键，之后对于keydown事件遍历比对SHIFT数组，根据当前是否有shift返回对应字符串。

![image-20220902173643678](/home/summage/Course/pict/ysyx/image-20220902173643678.png)

将取得的字符串传递给term->keypress函数完成对应处理。若返回不为空(即屏幕上会出现字符的按键类型)，则调用指令处理以及打印下一个sh>.（感觉没实现完整）

### 批处理系统

SYS_execve,SYS_exit修改较为简单。

> execve(const char \*pathname, char \* const argv[], char \* const envp[])
>
> 根据路径名使用新程序替代当前运行的程序，并初始化堆栈以及数据段（bss不初始化）。
>
> argv为命令行参数,且第一个应为文件名并以NULL作为结尾.
>
> envp则是作为新程序的环境变量,同样以NULL结尾.



















