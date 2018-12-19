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

	"gonum.org/v1/gonum/mat"

	"github.com/flyingyizi/tfutil/unsupervisedLearning/iforest"

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

type coll struct {
	Str  *string
	Data []float64
}

func main() {
	//flag
	filename := flag.String("f", "", "file stores the logs, default is outputing to terminal")
	num := flag.Uint64("n", math.MaxUint64, "number times of monitor/supervior")
	supervior := flag.Bool("s", false, "supervior the machine with trained model")
	interval := flag.Uint("interval", 250, "monitor interval,unit is milli-second")
	flag.Usage = usage

	flag.Parse()

	//register control+c
	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt, os.Kill)

	var wg sync.WaitGroup

	queue := make(chan coll, 100)

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

	//
	var forest *iforest.Forest
	if *supervior {
		forest = iforest.NewForest()
		if err := forest.Load("saved.txt.json"); err != nil {
			fmt.Println(err)
			return
		}
		fmt.Printf("load inforest trained model success, wit anomalBound:%v\n", forest.AnomalyBound)
	}

	//start log writing routine
	wg.Add(1)
	go writeFile(&wg, queue, w, forest)
	wg.Add(1)
	go producer(&wg, queue, c, *interval, *num)
	//start

	wg.Wait()

}

func producer(wg *sync.WaitGroup, queue chan coll, quit chan os.Signal, interval uint, maxrecord uint64) {

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
		if i > maxrecord {
			break
		}
		s, d := collet()
		queue <- coll{Str: s, Data: d}
		time.Sleep(time.Duration(interval) * time.Millisecond)
		i++
	}
ForEND:

	close(queue)
	//fmt.Println("producer | close channel, exit")
}

func writeFile(wg *sync.WaitGroup, queue chan coll, bufferedWriter *bufio.Writer, forest *iforest.Forest) {
	defer wg.Done()
	for msg := range queue {
		str, data := msg.Str, msg.Data

		//do predict
		if forest != nil {
			label, score, err := predict(data[1:], forest)
			if label == 1 && err == nil {
				fmt.Printf("time:%v , score:%v\n", time.Unix(0, int64(data[0])), score)
			}
			continue
		}
		//do record log
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

func predict(x []float64, forest *iforest.Forest) (label int, score float64, err error) {
	a := mat.NewDense(len(x), 1, x)
	labels, scores, e := forest.Predict(a)
	if e == nil {
		label, score = labels[0], scores[0]
	} else {
		err = e
	}
	return
}

func collet() (*string, []float64) {

	connections, _ := net.Connections("tcp")
	//c := cpuInfo()
	m, n := memInfo(), netInfo()

	cc, _ := cpu.Percent(0, false)

	// cpuPercent , memUsedPercent, recB,sendB, num conn
	d := make([]float64, 6)
	d[0] = float64(time.Now().UnixNano())
	d[1] = cc[0]
	d[2] = m.UsedPercent
	d[3] = float64(n.BytesRecv)
	d[4] = float64(n.BytesSent)
	d[5] = float64(len(connections))

	s := fmt.Sprintf("%v,  %v,%v,  %v,%v, %v\n", d[0], d[1], d[2], d[3], d[4], d[5])
	// s := fmt.Sprintf("%v,  %0.4f,%0.4f,  %v,%v, %v\n",
	// time.Now().UnixNano(),
	// cc[0], m.UsedPercent,
	// n.BytesRecv, n.BytesSent,
	// (len(connections)))

	return &s, d

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
