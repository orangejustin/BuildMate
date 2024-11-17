import { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import dayjs from 'dayjs';
import classNames from 'classnames';

const ChatMessage = ({ message, isUser }) => (
  <div className={classNames(
    "flex w-full mb-4",
    isUser ? "justify-end" : "justify-start"
  )}>
    <div className={classNames(
      "max-w-[70%] rounded-lg px-4 py-2",
      isUser ? "bg-user-message" : "bg-message-bg"
    )}>
      <p className="text-white">{message.content}</p>
      <span className="text-xs text-gray-400">
        {dayjs(message.createTime).format('HH:mm')}
      </span>
    </div>
  </div>
);

export default function ChatInterface() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const messagesEndRef = useRef(null);
  
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage = {
      role: 'user',
      content: input,
      createTime: Date.now(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');

    try {
      const response = await axios.post('http://localhost:8000/chat', {
        messages: [...messages, userMessage],
      });

      setMessages(prev => [...prev, response.data]);
    } catch (error) {
      console.error('Error:', error);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-chat-bg">
      <div className="flex-1 overflow-y-auto p-4">
        {messages.map((message, index) => (
          <ChatMessage 
            key={index}
            message={message}
            isUser={message.role === 'user'}
          />
        ))}
        <div ref={messagesEndRef} />
      </div>
      
      <form onSubmit={handleSubmit} className="p-4 border-t border-gray-700">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            className="flex-1 bg-message-bg text-white rounded-lg px-4 py-2 focus:outline-none"
            placeholder="Type a message..."
          />
          <button
            type="submit"
            className="bg-user-message text-white px-6 py-2 rounded-lg"
          >
            Send
          </button>
        </div>
      </form>
    </div>
  );
} 