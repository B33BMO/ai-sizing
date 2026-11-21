/* simd: stand-in for a long-running process in the judgment-eval server image.
 * Copied/exec'd under any name (argv[0] sets comm, so ps/lsof show the right command). Driven by env:
 *   SIM_PORT=8080            listen on 0.0.0.0:PORT (ss/lsof show it)
 *   SIM_HOLD=/path           keep this file open (lsof shows it)
 *   SIM_GROW=/path           append 64 KiB per tick, up to SIM_GROW_MAX bytes (an upload in progress)
 *   SIM_LOG=/path SIM_MSG="... %d%% ..."  append a progress line every SIM_EVERY ticks (percent goes up)
 *   SIM_EXIT_AFTER=3600      exit after that many seconds (as the box's init: the box's lifetime)
 */
#include <arpa/inet.h>
#include <fcntl.h>
#include <libgen.h>
#include <netinet/in.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

int main(int argc, char **argv) {
    char name[64];
    strncpy(name, argv[0], sizeof name - 1); name[sizeof name - 1] = 0;
    prctl(PR_SET_NAME, basename(name));
    signal(SIGHUP, SIG_IGN);
    signal(SIGCHLD, SIG_IGN);  /* also serves as the box's PID 1 (/sbin/init): reap orphans */
    const char *port = getenv("SIM_PORT"), *hold = getenv("SIM_HOLD"), *grow = getenv("SIM_GROW"),
               *log = getenv("SIM_LOG"), *msg = getenv("SIM_MSG"), *every = getenv("SIM_EVERY"),
               *gmax = getenv("SIM_GROW_MAX"), *pct0 = getenv("SIM_PCT"), *life = getenv("SIM_EXIT_AFTER");
    if (port) {
        int s = socket(AF_INET, SOCK_STREAM, 0), one = 1;
        setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
        struct sockaddr_in a = {.sin_family = AF_INET, .sin_port = htons(atoi(port)), .sin_addr.s_addr = INADDR_ANY};
        if (bind(s, (struct sockaddr *)&a, sizeof a) == 0) listen(s, 16);
    }
    if (hold) open(hold, O_RDWR | O_CREAT, 0640);
    int gfd = grow ? open(grow, O_WRONLY | O_CREAT | O_APPEND, 0640) : -1;
    long gsize = 0, gcap = gmax ? atol(gmax) : 8L << 20;
    int ev = every ? atoi(every) : 5, pct = pct0 ? atoi(pct0) : 40;
    static char chunk[65536];
    memset(chunk, 'x', sizeof chunk);
    for (long tick = 1;; tick++) {
        sleep(1);
        if (life && tick >= atol(life)) return 0;
        if (gfd >= 0 && gsize < gcap) gsize += write(gfd, chunk, sizeof chunk);
        if (log && msg && tick % ev == 0) {
            FILE *f = fopen(log, "a");
            if (f) {
                char ts[32]; time_t t = time(0);
                strftime(ts, sizeof ts, "%Y-%m-%d %H:%M:%S", localtime(&t));
                fprintf(f, "%s ", ts); fprintf(f, msg, pct < 99 ? ++pct : 99); fputc('\n', f); fclose(f);
            }
        }
    }
}
