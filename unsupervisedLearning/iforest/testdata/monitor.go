package main

import (
	"bufio"
	"flag"
	"fmt"
	"math"
	"os"
	"os/signal"
	"path"
	"sync"
	"time"

	"github.com/shirou/gopsutil/cpu"

	"github.com/shirou/gopsutil/mem"
	"github.com/shirou/gopsutil/net"
)

type Task struct {
	Wg *sync.WaitGroup
	//quit  chan bool
	Queue chan *string
}

func usage() {
	fmt.Fprintf(os.Stderr, `output machine monitor information
info: time, cpuPercent , memUsedPercent, net recB,sendB, num of conn
	
Options:
`)
	flag.PrintDefaults()
}

func main() {
	//flag
	filename := flag.String("file", "", "file stores the logs, default is outputing to terminal")
	num := flag.Uint64("num", math.MaxUint64, "number of log record")
	interval := flag.Uint("interval", 250, "monitor interval,unit is milli-second")
	flag.Usage = usage

	flag.Parse()

	//register control+c
	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt, os.Kill)

	var wg sync.WaitGroup

	queue := make(chan *string, 100)

	var w *bufio.Writer
	if *filename != "" {
		// if file exist rename it
		_, err := os.Stat(*filename)
		if err != nil && os.IsNotExist(err) {
		} else {
			//rename exist file
			dir, ff := path.Split(*filename)
			nf := time.Now().Format("2006-01-02-15-04-05") + "-back-" + ff
			os.Rename(*filename, path.Join(dir, nf))
		}
		f, err := os.OpenFile(*filename, os.O_WRONLY|os.O_CREATE, 0666)
		if err != nil {
			fmt.Println(err)
			return
		}
		w = bufio.NewWriter(f)
		defer f.Close()
	}

	//start log writing routine
	wg.Add(1)
	go writeFile(&wg, queue, w)
	wg.Add(1)
	go producer(&wg, queue, c, *interval, *num)
	//start

	wg.Wait()

}

func producer(wg *sync.WaitGroup, queue chan *string, quit chan os.Signal, interval uint, maxrecord uint64) {

	defer wg.Done()
	var i uint64
	for {
		select {
		case s := <-quit:
			fmt.Println()
			fmt.Println("producer | get", s)
			goto ForEND
		default:
		}

		//
		s := collet()
		queue <- s
		i++
		if i > maxrecord-1 {
			break
		}
		time.Sleep(time.Duration(interval) * time.Millisecond)
	}
ForEND:

	close(queue)
	//fmt.Println("producer | close channel, exit")
}

func writeFile(wg *sync.WaitGroup, queue chan *string, bufferedWriter *bufio.Writer) {
	defer wg.Done()
	for str := range queue {
		if str != nil {
			if bufferedWriter == nil {
				fmt.Print(*str)
			} else if _, err := bufferedWriter.WriteString(*str); err != nil {
				fmt.Println(err)
			}
		}
	}
	//go to here means queue is closed
	if bufferedWriter != nil {
		bufferedWriter.Flush()
	}
	//fmt.Println("Consumer | exit")
}

func collet() *string {

	connections, _ := net.Connections("tcp")
	//c := cpuInfo()
	m, n := memInfo(), netInfo()

	cc, _ := cpu.Percent(0, false)

	// cpuPercent , memUsedPercent, recB,sendB, num conn
	s := fmt.Sprintf("%v,  %0.4f,%0.4f,  %v,%v, %v\n",
		time.Now().UnixNano(),
		cc[0], m.UsedPercent,
		n.BytesRecv, n.BytesSent,
		(len(connections)))

	return &s

}

//cpuInfo  summary information about cpu (not each cpus)
func cpuInfo() *cpu.TimesStat {
	// （1）CPU信息
	// 操作系统的CPU利用率有以下几个部分：
	// User Time，执行用户进程的时间百分比；
	// System Time，执行内核进程和中断的时间百分比；
	// Wait IO，由于IO等待而使CPU处于idle（空闲）状态的时间百分比；
	// Idle，CPU处于idle状态的时间百分比。
	// UsedPercent := func() string {
	// 	cc, _ := cpu.Percent(time.Second, false)
	// 	return fmt.Sprintf("%s, %0.4f", cc[0]) //append CPU Used percent
	// }

	//cpu-total
	c, _ := cpu.Times(false)
	return &(c[0])
}

//memInfo summary information about memery
func memInfo() *mem.VirtualMemoryStat {
	// 内存利用率信息涉及
	// total（内存总数）、used（已使用的内存数）、
	// free（空闲内存数）、buffers（缓冲使用数）、cache（缓存使用数）
	v, _ := mem.VirtualMemory()
	return v
}

//swapMemInfo summary information about swap memory
func swapMemInfo() *mem.SwapMemoryStat {
	v, _ := mem.SwapMemory()
	return v
}

//netInfo summary information about net (not each netI/O)
func netInfo() *net.IOCountersStat {
	// 系统的网络信息与磁盘IO类似，涉及几个关键点，包括bytes_sent（发送字节数）、
	// bytes_recv=28220119（接收字节数）、packets_sent=200978（发送数据包数）、
	// packets_recv=212672（接收数据包数）等。
	// #pernic=True输出每个网络接口的IO信息, =false 输出汇总信息
	v, _ := net.IOCounters(false)
	return &(v[0])
	//{"name":"all","bytesSent":3737477,"bytesRecv":12211396,"packetsSent":19106,"packetsRecv":19595,"errin":0,"errout":0,"dropin":0,"dropout":0,"fifoin":0,"fifoout":0}
}

func diskInfo() {
	// 磁盘IO信息包括read_count（读IO数）、write_count（写IO数）、
	// read_bytes（IO读字节数）、write_bytes（IO写字节数）、
	// read_time（磁盘读时间）、write_time（磁盘写时间）等

	// v, _ := disk.IOCounters()
	// for k, v := range v {
	// 	fmt.Println(k, ":", v.String())
	// }
	//C: : {"readCount":0,"mergedReadCount":0,"writeCount":0,"mergedWriteCount":0,"readBytes":0,"writeBytes":0,"readTime":0,"writeTime":0,"iopsInProgress":0,"ioTime":0,"weightedIO":0,"name":"C:","serialNumber":"","label":""}
}
