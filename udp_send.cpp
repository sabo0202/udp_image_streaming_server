#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
int main()
{
 	int sock;
 	struct sockaddr_in addr;
 	int n;
 	
 	/* if (argc != 2) {
 	 	fprintf(stderr, "Usage : %s dstipaddr\n", argv[0]);
 	 	return 1;
 	} */
 	
 	sock = socket(AF_INET, SOCK_DGRAM, 0);
 	
 	addr.sin_family = AF_INET;
 	addr.sin_port = htons(12345);
	// addr.sin_addr.s_addr = inet_addr("192.168.1.1");
        // addr.sin_addr.s_addr = inet_addr("10.10.3.228");
        addr.sin_addr.s_addr = inet_addr("10.10.6.62");

 	// inet_pton(AF_INET, argv[1], &addr.sin_addr.s_addr);
 	
 	n = sendto(sock, "HELLO", 5, 0, (struct sockaddr *)&addr, sizeof(addr));
 	if (n < 1) {
 	 	perror("sendto");
 	 	return 1;
 	}
 	
 	close(sock);
 	
 	return 0;
}
