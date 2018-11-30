package main

import (
	"bufio"
	"fmt"
	"os"
	"os/signal"
	"sync"
	"time"

	"github.com/shirou/gopsutil/cpu"

	"github.com/shirou/gopsutil/mem"
)

type Task struct {
	Wg *sync.WaitGroup
	//quit  chan bool
	Queue chan *string
}

func main() {
	//register control+c
	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt, os.Kill)

	var wg sync.WaitGroup

	queue := make(chan *string, 100)

	filename := "test.txt"
	// if file exist rename it
	_, err := os.Stat("test.txt")
	if err != nil {
		if os.IsNotExist(err) {
			//log.Fatal("File does not exist.")
		}
	} else {
		//rename exist file
		err := os.Rename(filename, "back"+filename)
		if err != nil {
			//log.Fatal(err)
		}
	}

	//task := Task{Wg: &wg /*  quit: make(chan bool), */, Queue: queue}
	//start log writing routine
	wg.Add(1)
	go WriteFile(&wg, queue, filename)
	wg.Add(1)
	go Producer(&wg, queue, c)
	//start

	wg.Wait()

}

func Producer(wg *sync.WaitGroup, queue chan *string, quit chan os.Signal) {
	defer wg.Done()
	i := 0
	for {
		select {
		case s := <-quit:
			fmt.Println()
			fmt.Println("Producer | get", s)
			goto ForEND
		default:
		}

		//
		s := collet()
		queue <- s
		i++
		if i > 10000 {
			break
		}
		time.Sleep(500 * time.Millisecond)
	}
ForEND:

	close(queue)
	fmt.Println("Producer | close channel, exit")
}

func WriteFile(wg *sync.WaitGroup, queue chan *string, filename string) {
	defer wg.Done()

	file, err := os.OpenFile(filename, os.O_WRONLY|os.O_CREATE, 0666)
	if err != nil {
		//log.Fatal(err)
		return
	}
	defer file.Close()
	// create buffered writer for the file
	bufferedWriter := bufio.NewWriter(file)

	for str := range queue {
		if str != nil {
			if _, err := bufferedWriter.WriteString(*str); err != nil {
				fmt.Println(err)
			}
			//fmt.Print("consumer | get", *str)
		}
	}
	//go to here means queue is closed
	bufferedWriter.Flush()
	fmt.Println("Consumer | channel closed")

	//fmt.Println("Consumer | exit")

	// if err != nil {
	// 	log.Fatal(err)
	// }
	// log.Printf("Bytes written: %d\n", bytesWritten)
	// // 写字符串到buffer
	// // 也可以使用 WriteRune() 和 WriteByte()
	// bytesWritten, err = bufferedWriter.WriteString(
	// 	"Buffered string\n",
	// )
	// if err != nil {
	// 	log.Fatal(err)
	// }
	// log.Printf("Bytes written: %d\n", bytesWritten)
	// // 检查缓存中的字节数
	// unflushedBufferSize := bufferedWriter.Buffered()
	// log.Printf("Bytes buffered: %d\n", unflushedBufferSize)
	// // 还有多少字节可用（未使用的缓存大小）
	// bytesAvailable := bufferedWriter.Available()
	// if err != nil {
	// 	log.Fatal(err)
	// }
	// log.Printf("Available buffer: %d\n", bytesAvailable)
	// // 写内存buffer到硬盘
	// bufferedWriter.Flush()
	// // 丢弃还没有flush的缓存的内容，清除错误并把它的输出传给参数中的writer
	// // 当你想将缓存传给另外一个writer时有用
	// bufferedWriter.Reset(bufferedWriter)
	// bytesAvailable = bufferedWriter.Available()
	// if err != nil {
	// 	log.Fatal(err)
	// }
	// log.Printf("Available buffer: %d\n", bytesAvailable)
	// // 重新设置缓存的大小。
	// // 第一个参数是缓存应该输出到哪里，这个例子中我们使用相同的writer。
	// // 如果我们设置的新的大小小于第一个参数writer的缓存大小， 比如10，我们不会得到一个10字节大小的缓存，
	// // 而是writer的原始大小的缓存，默认是4096。
	// // 它的功能主要还是为了扩容。
	// bufferedWriter = bufio.NewWriterSize(
	// 	bufferedWriter,
	// 	8000,
	// )
	// // resize后检查缓存的大小
	// bytesAvailable = bufferedWriter.Available()
	// if err != nil {
	// 	log.Fatal(err)
	// }
	// log.Printf("Available buffer: %d\n", bytesAvailable)
}

func collet() *string {
	v, _ := mem.VirtualMemory()
	s := fmt.Sprintf("%v, %v, %0.4f", v.Active, v.Free, v.UsedPercent)

	cc, _ := cpu.Percent(time.Second, false)
	s = fmt.Sprintf("%s, %0.4f\n", s, cc[0]) //append CPU Used percent

	return &s

	// c, _ := cpu.Info()
	// d, _ := disk.Usage("/")
	// n, _ := host.Info()
	// net.ConnectionsMax
	// nv, _ := net.IOCounters(true)
	// boottime, _ := host.BootTime()
	// btime := time.Unix(int64(boottime), 0).Format("2006-01-02 15:04:05")
	// fmt.Printf("        Mem       : %v MB  Free: %v MB Used:%v Usage:%f%%\n", v.Total/1024/1024, v.Available/1024/1024, v.Used/1024/1024, v.UsedPercent)
	// if len(c) > 1 {
	// 	for _, sub_cpu := range c {
	// 		modelname := sub_cpu.ModelName
	// 		cores := sub_cpu.Cores
	// 		fmt.Printf("        CPU       : %v   %v cores \n", modelname, cores)
	// 	}
	// } else {

	// 	sub_cpu := c[0]
	// 	modelname := sub_cpu.ModelName
	// 	cores := sub_cpu.Cores
	// 	fmt.Printf("        CPU       : %v   %v cores \n", modelname, cores)

	// }

	// fmt.Printf("        Network: %v bytes / %v bytes\n", nv[0].BytesRecv, nv[0].BytesSent)
	// fmt.Printf("        SystemBoot:%v\n", btime)
	// fmt.Printf("        CPU Used    : used %f%% \n", cc[0])
	// fmt.Printf("        HD        : %v GB  Free: %v GB Usage:%f%%\n", d.Total/1024/1024/1024, d.Free/1024/1024/1024, d.UsedPercent)
	// fmt.Printf("        OS        : %v(%v)   %v  \n", n.Platform, n.PlatformFamily, n.PlatformVersion)
	// fmt.Printf("        Hostname  : %v  \n", n.Hostname)
}
