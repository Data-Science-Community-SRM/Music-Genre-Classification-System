
// ----------------------------------------- Name of the File -------------


window.onload = () => {
    function getFileListIds() {
      const fileListUl = document.getElementById("fileList");
      let children = fileListUl.childNodes;
      let ids = [];
      children.forEach((child) => ids.push(child.id));
      return ids.map(Number);
    }
  
    const audioElement = document.getElementById("audio");
  
    audioElement.onended = function () {
      playNext();
    };
    const fileElement = document.getElementById("fileupload");
    const fileListUl = document.getElementById("fileList");
    let loadFileList;
    let playList = [];
    let nowPlayingLoadOrderIndex = 0;

    
  // ---------------------------------------- Analyser for Audio ------------------------------------

    const analyserDiv = document.getElementById("analyserOut");
    let audioCtx;
    let source;
    let analyser;
  
    function playFile(file) {
      audioElement.src = URL.createObjectURL(file);
      audioElement.load();
      audioElement.play();
    }
  
    function playFileViaLoadIndex(i) {
      let file = loadFileList[i];
      nowPlayingLoadOrderIndex = i;
      for (let j = 0; j < loadFileList.length; j++) {
        document.getElementById(`${j}`).style.color = "black";
      }
      document.getElementById(`${i}`).style.color = "transparent";
      playFile(file);
    }
  
    function playNext() {
      let currPlayingIndex = playList.findIndex(
        (item) => item.loadOrder === nowPlayingLoadOrderIndex
      );
  
      if (loadFileList && currPlayingIndex < loadFileList.length - 1) {
        let nextFile = playList[currPlayingIndex + 1];
        const nextIndex = nextFile.loadOrder;
        playFileViaLoadIndex(nextIndex);
      } else if (!!loadFileList) {
        let nextFile = playList[0];
        const nextIndex = nextFile.loadOrder;
        playFileViaLoadIndex(nextIndex);
      }
    }
  
    fileElement.onchange = function () {
      loadFileList = this.files;
      let fl = loadFileList.length;
      let i = 0;
      fileListUl.innerHTML = "";
  
      while (i < fl) {
        // localize file var in the loop
        let file = loadFileList[i];
        let fileName = file.name;
        playList.push({
          loadOrder: i,
          fileName
        });
  
        const newListItem = document.createElement("li");
        newListItem.id = `${i}`;
        const newContent = document.createTextNode(fileName);
        newListItem.appendChild(newContent);
        fileListUl.appendChild(newListItem);
  
        i++;
      }
      enableDragSort("drag-sort-enable");
  
      audioCtx = new AudioContext();
      if (!source) source = audioCtx.createMediaElementSource(audioElement);
      if (!analyser) analyser = audioCtx.createAnalyser();
  
      playFileViaLoadIndex(0);
  
      source.connect(analyser);
      analyser.connect(audioCtx.destination);
      analyser.fftSize = 32;
      var bufferLength = analyser.frequencyBinCount;
      var dataArray = new Uint8Array(bufferLength);
  
      function log() {
        window.requestAnimationFrame(log);
        analyser.getByteFrequencyData(dataArray);
        const output = [];
        dataArray.forEach((number) => {
          output.push(`<div>${Array(number).fill("|").join("")}</div>`);
        });
        analyserDiv.innerHTML = output.join("");
      }
      log();
    };
  
    
    function enableDragSort(listClass) {
      const sortableLists = document.getElementsByClassName(listClass);
      Array.prototype.map.call(sortableLists, (list) => {
        enableDragList(list);
      });
    }
  
    function enableDragList(list) {
      Array.prototype.map.call(list.children, (item) => {
        enableDragItem(item);
        enableDoubleClickItem(item);
      });
    }
  
    function enableDoubleClickItem(item) {
      item.ondblclick = handleDblClick;
    }
  
    function handleDblClick(item) {
      const loadIndexId = Number(item.target.id);
      playFileViaLoadIndex(loadIndexId);
    }
  
    function enableDragItem(item) {
      item.setAttribute("draggable", true);
      item.ondrag = handleDrag;
      item.ondragover = (e) => e.preventDefault();
      item.ondragend = handleDrop;
    }
  
    function handleDrag(item) {
      const selectedItem = item.target,
        list = selectedItem.parentNode,
        x = event.clientX,
        y = event.clientY;
  
      selectedItem.classList.add("drag-sort-active");
      let swapItem =
        document.elementFromPoint(x, y) === null
          ? selectedItem
          : document.elementFromPoint(x, y);
  
      if (list === swapItem.parentNode) {
        swapItem =
          swapItem !== selectedItem.nextSibling ? swapItem : swapItem.nextSibling;
        list.insertBefore(selectedItem, swapItem);
      }
    }
  
    function handleDrop(item) {
      item.target.classList.remove("drag-sort-active");
      let ids = getFileListIds();
      let newPlayList = ids.map((id) =>
        playList.find((item) => item.loadOrder === id)
      );
      playList = newPlayList;
    }
  };
  